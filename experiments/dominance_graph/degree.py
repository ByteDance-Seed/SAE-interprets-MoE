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
Section 4.2 and Appendix C: degree distributions of the dominance graph.

Expert j dominates latent i when s(j, i) >= tau. Reported per layer:

- latent side: the share of latents dominated by exactly 0, 1, 2, ... experts (Figure 6). Since
  the s(., i) sum to one, a latent has fewer than 1 / tau + 1 dominators.
- expert side: the number of latents every expert dominates, sorted in ascending order (Figure 7),
  with its minimum, mean, median and the share of all edges held by the ten largest experts.

Input: the aggregated contribution statistics of each layer.
Output: {output_dir}/tau_{tau}/degree.json, also printed.
'''

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import dominance_adjacency, save_json
from common.paths import LAYERS, OUTPUT_DIR, aggregated_path

_N_TOP_EXPERTS = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type = int, nargs = '+', default = list(LAYERS))
    p.add_argument('--tau', type = float, default = 0.2)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'dominance_graph'))
    return p.parse_args()


def layer_degree(layer, tau):
    adjacency_exf = dominance_adjacency(aggregated_path(layer), tau)
    with safe_open(str(aggregated_path(layer)), framework = 'pt') as f:
        act_count_f = f.get_tensor('act_count')
    n_experts, n_latents = adjacency_exf.shape

    latent_degree_hist = torch.bincount(adjacency_exf.sum(dim = 0), minlength = int(1 / tau) + 1)
    expert_degree_e = adjacency_exf.sum(dim = 1).sort().values
    n_edges = int(expert_degree_e.sum())
    return {
        'n_latents': n_latents,
        'n_latents_never_active': int((act_count_f == 0).sum()),
        'latent_degree_count': latent_degree_hist.tolist(),
        'latent_degree_fraction': (latent_degree_hist.double() / n_latents).tolist(),
        'n_edges': n_edges,
        'expert_degree_sorted': expert_degree_e.tolist(),
        'expert_degree_min': int(expert_degree_e[0]),
        'expert_degree_mean': n_edges / n_experts,
        'expert_degree_median': float(torch.quantile(expert_degree_e.double(), 0.5)),
        'top10_edge_share': int(expert_degree_e[-_N_TOP_EXPERTS:].sum()) / n_edges,
        'top10_edge_share_if_uniform': _N_TOP_EXPERTS / n_experts,
    }


def main():
    args = parse_args()
    result = {'tau': args.tau, 'layers': {}}
    for layer in args.layers:
        degree = layer_degree(layer, args.tau)
        result['layers'][str(layer)] = degree
        print(f'\n=== layer {layer}, tau {args.tau} ===')
        print('latents by number of dominators: ' + '  '.join(
            f'{d}: {v * 100:.3f}%' for d, v in enumerate(degree['latent_degree_fraction'])
        ))
        print(f"latents never active: {degree['n_latents_never_active']}")
        print(f"expert degree: min {degree['expert_degree_min']}, mean {degree['expert_degree_mean']:.0f}, "
              f"median {degree['expert_degree_median']:.0f}, "
              f"top {_N_TOP_EXPERTS} experts hold {degree['top10_edge_share'] * 100:.1f}% of the edges "
              f"({degree['top10_edge_share_if_uniform'] * 100:.1f}% if uniform)")

    save_json(result, Path(args.output_dir) / f'tau_{args.tau}' / 'degree.json')


if __name__ == '__main__':
    main()
