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
The released SAEs are TopK SAEs (k = 100) trained on the MoE block output of Qwen3-30B-A3B.

Hugging Face layout (repo SAE_REPO), one directory per layer:

    layer_{4,30,46}/sae.safetensors   the four tensors below, float32
    layer_{4,30,46}/config.json      layer, architecture, k and tensor shapes

    W_enc  [d_model, d_latent]    encoder weights, column i belongs to latent i
    b_enc  [d_latent]
    W_dec  [d_latent, d_model]    decoder weights, row i has unit L2 norm
    b_dec  [d_model]

with d_model = 2048 and d_latent = 2^20. For a raw MoE output y, the SAE computes
TopK_100(ReLU(y @ W_enc + b_enc)) and decodes as acts @ W_dec + b_dec; the encoder does not
subtract b_dec. The activation normalization factor used in training is already folded into the
weights, so the SAE applies directly to the raw layer output.
'''

from huggingface_hub import hf_hub_download
from safetensors import safe_open

SAE_REPO = 'ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE'
SAE_FILENAME = 'layer_{layer}/sae.safetensors'
ACT_K = 100


def resolve_sae_path(layer, sae_path = None, repo = SAE_REPO):
    if sae_path is not None:
        return sae_path
    return hf_hub_download(repo, SAE_FILENAME.format(layer = layer))


def load_encoder(sae_path, device):
    '''Returns W_enc as [d_latent, d_model], so that rows of active latents can be gathered directly.'''
    with safe_open(sae_path, framework = 'pt', device = str(device)) as f:
        W_enc_hxf = f.get_tensor('W_enc').float()
        b_enc_f = f.get_tensor('b_enc').float()
    W_enc_fxh = W_enc_hxf.T.contiguous()
    del W_enc_hxf
    return W_enc_fxh, b_enc_f
