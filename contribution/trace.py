# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Compute the expert-to-latent contributions of Section 3 on one slice of a parquet file, online
during the forward pass of Qwen3-30B-A3B.

For every hooked MoE layer and every token t, with activated experts A(t), gating weights g_j(t)
and expert outputs u_j(t), the layer output y(t) is encoded by the SAE. For every activated
latent i (TopK selected with a positive activation),

    c(j, i, t) = ReLU(g_j(t) * W_enc,(i) . u_j(t))                    Eq. (6)
    C(j, i, t) = c(j, i, t) / sum_{j' in A(t)} c(j', i, t)            Eq. (7)

and C is summed over tokens per (expert, latent). Dividing by the number of tokens gives Eq. (8),
and the normalization over experts in Eq. (9) is done by contribution/aggregate.py, so partial
results from many jobs simply add up.

The gating weights and the expert outputs are recomputed exactly as Qwen3MoeSparseMoeBlock does
it, including the cast of the renormalized top-k weights back to the model dtype, so the numbers
match the paper's pipeline, which stored them in bf16 before tracing.

Accumulated per layer, all of them sums over the processed tokens:

    C                 [n_experts, d_latent]  float32  sum_t C(j, i, t)
    act_count         [d_latent]             int64    number of tokens on which latent i is active
    gating_rank_sum   [k]                    float64  gating weights sorted in descending order
    contrib_rank_sum  [k]                    float64  C(., i, t) sorted in descending order, over
                                                      (token, active latent) pairs
    n_positive_hist   [k + 1]                int64    number of experts with c > 0 per pair
    routing_mass      [n_experts, vocab]     float64  gating mass expert j receives on token id v
    n_tokens, n_pairs [1]                    int64

A pair whose k contributions are all zero (the latent is activated by the encoder bias alone) has
no defined C. It counts in act_count but not in n_pairs or the rank profile.

Output: {output_dir}/layer{layer}/parts/{parquet stem}_{idx_start}-{idx_end}.safetensors.
An existing output is skipped, so rerunning the same command resumes a preempted job.

Suffix convention for tensor names: n tokens of a document, b tokens of a chunk, h hidden size,
e activated experts per token, te all experts, f SAE latents, k active latents per token,
v vocabulary.
'''

import argparse
import sys
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import save_safetensors
from common.paths import MODEL_NAME, OUTPUT_DIR
from common.sae import ACT_K, SAE_REPO, load_encoder, resolve_sae_path

_EPS = 1e-9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type = str, default = MODEL_NAME)
    p.add_argument('--layers', type = int, nargs = '+', required = True)
    p.add_argument('--sae_paths', type = str, nargs = '+', default = None,
                   help = f'one local SAE file per layer; downloaded from {SAE_REPO} when omitted')
    p.add_argument('--parquet_path', type = str, required = True, help = 'parquet file with a "text" column')
    p.add_argument('--idx_start', type = int, default = 0)
    p.add_argument('--idx_end', type = int, default = None)
    p.add_argument('--max_length', type = int, default = 4096)
    p.add_argument('--act_k', type = int, default = ACT_K)
    p.add_argument('--chunk_size', type = int, default = 1024,
                   help = 'tokens encoded at once; the dense pre-activation takes chunk_size * 4 MiB')
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'contribution'))
    args = p.parse_args()
    if args.sae_paths is not None and len(args.sae_paths) != len(args.layers):
        p.error('--sae_paths needs exactly one file per layer')
    if len(set(args.layers)) != len(args.layers):
        p.error('--layers must be unique')
    return args


class LayerStats:

    def __init__(self, sae_path, n_experts, vocab_size, args, device):
        self.W_enc_fxh, self.b_enc_f = load_encoder(sae_path, device)
        d_latent = self.W_enc_fxh.shape[0]
        k = args.top_k
        kwargs = {'device': device}
        self.act_k = args.act_k
        self.chunk_size = args.chunk_size

        self.C_texf = torch.zeros(n_experts, d_latent, dtype = torch.float32, **kwargs)
        self.act_count_f = torch.zeros(d_latent, dtype = torch.int64, **kwargs)
        self.gating_rank_sum_e = torch.zeros(k, dtype = torch.float64, **kwargs)
        self.contrib_rank_sum_e = torch.zeros(k, dtype = torch.float64, **kwargs)
        self.n_positive_hist = torch.zeros(k + 1, dtype = torch.int64, **kwargs)
        self.routing_mass_texv = torch.zeros(n_experts, vocab_size, dtype = torch.float64, **kwargs)
        self.n_tokens = torch.zeros(1, dtype = torch.int64, **kwargs)
        self.n_pairs = torch.zeros(1, dtype = torch.int64, **kwargs)

    def update(self, token_id_n, mlp_out_nxh, expert_out_nxexh, gating_nxe, expert_id_nxe):
        self.n_tokens += token_id_n.numel()
        self.routing_mass_texv.index_put_(
            (expert_id_nxe, token_id_n[:, None].expand_as(expert_id_nxe)),
            gating_nxe.to(torch.float64),
            accumulate = True,
        )
        for start in range(0, token_id_n.numel(), self.chunk_size):
            chunk = slice(start, start + self.chunk_size)
            self.update_chunk(mlp_out_nxh[chunk], expert_out_nxexh[chunk], gating_nxe[chunk], expert_id_nxe[chunk])

    def update_chunk(self, mlp_out_bxh, expert_out_bxexh, gating_bxe, expert_id_bxe):
        gating_bxe = gating_bxe.float()

        pre_acts_bxf = mlp_out_bxh.float() @ self.W_enc_fxh.T + self.b_enc_f
        act_bxk, latent_idx_bxk = torch.topk(torch.relu_(pre_acts_bxf), k = self.act_k, dim = -1)
        del pre_acts_bxf
        # TopK fills up with zeros when fewer than act_k latents are positive; those are not active
        active_bxk = act_bxk > 0
        self.act_count_f += torch.bincount(latent_idx_bxk[active_bxk], minlength = self.act_count_f.numel())

        W_bxkxh = self.W_enc_fxh[latent_idx_bxk]
        c_bxexk = torch.relu(torch.bmm(expert_out_bxexh.float(), W_bxkxh.transpose(1, 2)) * gating_bxe[:, :, None])
        del W_bxkxh
        denom_bxk = c_bxexk.sum(dim = 1)
        valid_bxk = active_bxk & (denom_bxk > 0)
        C_bxexk = c_bxexk / denom_bxk[:, None, :].clamp_min(_EPS) * valid_bxk[:, None, :]

        self.C_texf.index_put_(
            (expert_id_bxe[:, :, None].expand_as(C_bxexk), latent_idx_bxk[:, None, :].expand_as(C_bxexk)),
            C_bxexk,
            accumulate = True,
        )

        self.gating_rank_sum_e += torch.sort(gating_bxe, dim = -1, descending = True).values.sum(dim = 0).double()
        # invalid pairs are all zero, so they add nothing to the sorted sums
        self.contrib_rank_sum_e += torch.sort(C_bxexk, dim = 1, descending = True).values.sum(dim = (0, 2)).double()
        self.n_positive_hist += torch.bincount((c_bxexk > 0).sum(dim = 1)[valid_bxk],
                                               minlength = self.n_positive_hist.numel())
        self.n_pairs += valid_bxk.sum()

    def tensors(self):
        return {
            'C': self.C_texf,
            'act_count': self.act_count_f,
            'gating_rank_sum': self.gating_rank_sum_e,
            'contrib_rank_sum': self.contrib_rank_sum_e,
            'n_positive_hist': self.n_positive_hist,
            'routing_mass': self.routing_mass_texv,
            'n_tokens': self.n_tokens,
            'n_pairs': self.n_pairs,
        }


class _StopForward(Exception):
    '''Raised after the deepest hooked layer, the remaining layers are not needed.'''


def routing(moe_block, router_logits_nxte, dtype):
    '''Same computation as Qwen3MoeSparseMoeBlock.forward.'''
    weights_nxte = F.softmax(router_logits_nxte, dim = 1, dtype = torch.float)
    weights_nxe, expert_id_nxe = torch.topk(weights_nxte, moe_block.top_k, dim = -1)
    if moe_block.norm_topk_prob:
        weights_nxe /= weights_nxe.sum(dim = -1, keepdim = True)
    return weights_nxe.to(dtype), expert_id_nxe


def expert_outputs(moe_block, mlp_in_nxh, expert_id_nxe):
    '''u_j(t) for the activated experts, before multiplying by the gating weight.'''
    expert_out_nxexh = mlp_in_nxh.new_zeros(*expert_id_nxe.shape, mlp_in_nxh.shape[-1])
    for expert_id in expert_id_nxe.unique().tolist():
        token_idx, slot_idx = (expert_id_nxe == expert_id).nonzero(as_tuple = True)
        expert_out_nxexh[token_idx, slot_idx] = moe_block.experts[expert_id](mlp_in_nxh[token_idx])
    return expert_out_nxexh


def hook_fn(moe_block, inputs, outputs, stats, state, is_last):
    mlp_in_nxh = inputs[0].flatten(0, 1)
    mlp_out, router_logits_nxte = outputs
    mlp_out_nxh = mlp_out.flatten(0, 1)
    with torch.no_grad():
        gating_nxe, expert_id_nxe = routing(moe_block, router_logits_nxte, mlp_in_nxh.dtype)
        expert_out_nxexh = expert_outputs(moe_block, mlp_in_nxh, expert_id_nxe)
        stats.update(state['token_id'].to(mlp_in_nxh.device), mlp_out_nxh, expert_out_nxexh, gating_nxe, expert_id_nxe)
    if is_last:
        raise _StopForward


def main():
    args = parse_args()

    stem = Path(args.parquet_path).stem
    texts = pq.ParquetFile(args.parquet_path).read(columns = ['text']).column('text')
    idx_end = len(texts) if args.idx_end is None else min(args.idx_end, len(texts))
    texts = texts.slice(args.idx_start, max(0, idx_end - args.idx_start)).to_pylist()
    print(f'loaded {len(texts)} documents from rows [{args.idx_start}, {idx_end}) of {args.parquet_path}.')

    dest_paths = {
        layer: Path(args.output_dir) / f'layer{layer}' / 'parts' / f'{stem}_{args.idx_start}-{idx_end}.safetensors'
        for layer in args.layers
    }
    if all(path.exists() for path in dest_paths.values()):
        print('all layers finished, exit.')
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype = torch.bfloat16, device_map = 'auto')
    model.eval()
    args.top_k = model.config.num_experts_per_tok

    state = {}
    stats = {}
    for idx, layer in enumerate(args.layers):
        moe_block = model.model.layers[layer].mlp
        assert isinstance(moe_block.experts, torch.nn.ModuleList), \
            'expected the per-expert ModuleList of Qwen3MoeSparseMoeBlock, see requirements.txt for the transformers version'
        device = moe_block.gate.weight.device
        sae_path = resolve_sae_path(layer, None if args.sae_paths is None else args.sae_paths[idx])
        stats[layer] = LayerStats(sae_path, moe_block.num_experts, len(tokenizer), args, device)
        moe_block.register_forward_hook(partial(
            hook_fn, stats = stats[layer], state = state, is_last = layer == max(args.layers)
        ))

    first_device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for text in tqdm(texts):
            if not text:
                continue
            input_ids = tokenizer(text, return_tensors = 'pt', max_length = args.max_length, truncation = True).input_ids
            state['token_id'] = input_ids[0]
            try:
                model(input_ids = input_ids.to(first_device), use_cache = False)
            except _StopForward:
                pass

    for layer in args.layers:
        save_safetensors(stats[layer].tensors(), dest_paths[layer])


if __name__ == '__main__':
    main()
