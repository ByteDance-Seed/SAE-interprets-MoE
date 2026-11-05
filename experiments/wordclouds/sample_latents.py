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
Section 5.1: sample up to 2000 latents uniformly from the dominant set of every expert, taking the
whole set when it is smaller. The sampled latents are the ones sent to auto-interpretation.

For the released experts, the sample used in the paper is released with the explanations: the
"feat" field of data/layer30/explanations/e{expert}.jsonl lists exactly the latents drawn for that expert,
including those whose explanation came out missing or Uninterpretable. Rerunning this script
draws a different sample of the same size.

Input: the aggregated contribution statistics of the layer.
Output: {output_dir}/layer{layer}/e{expert}.txt, one latent per line written as {layer}-F{latent}
like the "feat" field of the explanation files.
'''

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import dominance_adjacency
from common.paths import OUTPUT_DIR, aggregated_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type = int, default = 30)
    p.add_argument('--tau', type = float, default = 0.2)
    p.add_argument('--sample_size', type = int, default = 2000)
    p.add_argument('--seed', type = int, default = 721)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'sampled_latents'))
    return p.parse_args()


def main():
    args = parse_args()
    adjacency_exf = dominance_adjacency(aggregated_path(args.layer), args.tau)
    generator = torch.Generator().manual_seed(args.seed)
    output_dir = Path(args.output_dir) / f'layer{args.layer}'
    output_dir.mkdir(parents = True, exist_ok = True)

    for expert_id, row_f in enumerate(adjacency_exf):
        dominant_set = row_f.nonzero().flatten()
        picked = torch.randperm(len(dominant_set), generator = generator)[:args.sample_size]
        sampled = dominant_set[picked].sort().values.tolist()
        with open(output_dir / f'e{expert_id}.txt', 'w') as f:
            f.writelines(f'{args.layer}-F{latent}\n' for latent in sampled)
        print(f'expert {expert_id}: sampled {len(sampled)} of {len(dominant_set)} dominated latents')

    print(f'write finished to {output_dir}.')


if __name__ == '__main__':
    main()
