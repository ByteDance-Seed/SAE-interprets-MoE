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
Section 4.1 and Figure 2: rank profiles of the gating weight and of the contribution.

For every token the k gating weights are sorted in descending order and the value at rank r is
averaged over all tokens, giving g_r. For every (token, active latent) pair the k values
C(j, i, t) are sorted in descending order and averaged over all pairs, giving C_r. Both profiles
sum to one. Also reported is the mean number of activated experts whose contribution is
truncated to zero by the ReLU of Eq. (6).

Input: the aggregated contribution statistics of each layer.
Output: {output_dir}/rank_profile.json, also printed as a table.
'''

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import save_json
from common.paths import LAYERS, OUTPUT_DIR, aggregated_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type = int, nargs = '+', default = list(LAYERS))
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'rank_profile'))
    return p.parse_args()


def layer_profile(layer):
    with safe_open(str(aggregated_path(layer)), framework = 'pt') as f:
        stats = {key: f.get_tensor(key) for key in
                 ('gating_rank_sum', 'contrib_rank_sum', 'n_positive_hist', 'n_tokens', 'n_pairs')}
    n_tokens, n_pairs = int(stats['n_tokens']), int(stats['n_pairs'])
    k = stats['gating_rank_sum'].numel()
    n_positive_fraction = stats['n_positive_hist'].double() / n_pairs
    mean_positive = float((n_positive_fraction * torch.arange(k + 1)).sum())
    return {
        'n_tokens': n_tokens,
        'n_pairs': n_pairs,
        'gating': (stats['gating_rank_sum'] / n_tokens).tolist(),
        'contribution': (stats['contrib_rank_sum'] / n_pairs).tolist(),
        'n_positive_fraction': n_positive_fraction.tolist(),
        'mean_truncated_experts': k - mean_positive,
    }


def main():
    args = parse_args()
    result = {str(layer): layer_profile(layer) for layer in args.layers}

    for layer, profile in result.items():
        print(f"\n=== layer {layer}: {profile['n_tokens']} tokens, {profile['n_pairs']} (token, active latent) pairs ===")
        print(f"{'rank':>5}{'gating':>12}{'contribution':>14}")
        for r, (g, c) in enumerate(zip(profile['gating'], profile['contribution'])):
            print(f'{r:>5}{g:>12.6f}{c:>14.6f}')
        print(f"activated experts truncated by the ReLU per pair: {profile['mean_truncated_experts']:.2f}")

    save_json(result, Path(args.output_dir) / 'rank_profile.json')


if __name__ == '__main__':
    main()
