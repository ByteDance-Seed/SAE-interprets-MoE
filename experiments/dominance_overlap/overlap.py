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
Section 4.3 and Table 1: pairwise overlap of the dominant sets.

With B the boolean [n_experts, n_latents] adjacency of the dominance graph, the intersection
matrix is B B^T, from which follow for every unordered expert pair

    J(j, j')        = |S(j) & S(j')| / |S(j) | S(j')|        Jaccard similarity
    overlap(j, j')  = |S(j) & S(j')| / min(|S(j)|, |S(j')|)  share of the smaller set

and for every expert the private fraction, the share of its dominant set that no other expert
dominates.

Reported per layer, upper block of Table 1: mean and median of J, the share of pairs sharing no
latent, and the share of the total intersection carried by the 10% of pairs with the largest J.
Lower block, over the pairs with J > 0.05: their number, the experts involved, how many of these
have a single partner, the largest connected component of the graph these pairs form, and the
largest J. Quoted in the text: mutual nearest-neighbour pairs by J, the median overlap of the
pairs above 0.05, and the mean private fraction.

Input: the aggregated contribution statistics of each layer.
Output: {output_dir}/tau_{tau}/summary.json, and per layer jaccard_{layer}.safetensors holding
the jaccard and intersection matrices and the expert degrees, used by plot_heatmap.py and by the
expert_domains experiment.

Suffix convention: e expert, m latent, p unordered expert pair.
'''

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import dominance_adjacency, save_json, save_safetensors
from common.paths import LAYERS, OUTPUT_DIR, aggregated_path

_TOP_PAIR_FRACTION = 0.1
_STRONG_PAIR_J = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type = int, nargs = '+', default = list(LAYERS))
    p.add_argument('--tau', type = float, default = 0.2)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'dominance_overlap'))
    return p.parse_args()


def overlap_matrices(adjacency_em):
    expert_degree_e = adjacency_em.sum(dim = 1)
    latent_degree_m = adjacency_em.sum(dim = 0)

    adjacency_float_em = adjacency_em.to(torch.float32)
    intersection_ee = (adjacency_float_em @ adjacency_float_em.T).round().to(torch.int64)
    del adjacency_float_em
    intersection_ee.fill_diagonal_(0)
    union_ee = expert_degree_e[:, None] + expert_degree_e[None, :] - intersection_ee
    jaccard_ee = intersection_ee.double() / union_ee.clamp_min(1)

    private_e = (adjacency_em & (latent_degree_m == 1)[None, :]).sum(dim = 1)
    private_fraction_e = private_e.double() / expert_degree_e.clamp_min(1)
    return jaccard_ee, intersection_ee, expert_degree_e, private_fraction_e


def largest_component(linked_ee):
    n_experts = linked_ee.shape[0]
    seen, largest = set(), 0
    for root in range(n_experts):
        if root in seen or not linked_ee[root].any():
            continue
        seen.add(root)
        stack, size = [root], 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in linked_ee[node].nonzero().flatten().tolist():
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        largest = max(largest, size)
    return largest


def overlap_stats(jaccard_ee, intersection_ee, expert_degree_e, private_fraction_e):
    n_experts = jaccard_ee.shape[0]
    row_p, col_p = torch.triu_indices(n_experts, n_experts, offset = 1)
    jaccard_p = jaccard_ee[row_p, col_p]
    intersection_p = intersection_ee[row_p, col_p].double()
    smaller_p = torch.minimum(expert_degree_e[row_p], expert_degree_e[col_p]).double()
    overlap_p = intersection_p / smaller_p.clamp_min(1)

    n_top = math.ceil(len(jaccard_p) * _TOP_PAIR_FRACTION)
    top_p = jaccard_p.argsort(descending = True)[:n_top]

    strong_p = jaccard_p > _STRONG_PAIR_J
    linked_ee = jaccard_ee > _STRONG_PAIR_J
    linked_ee.fill_diagonal_(False)
    n_partners_e = linked_ee.sum(dim = 1)

    nearest_e = jaccard_ee.argmax(dim = 1)
    n_mutual = sum(1 for j in range(n_experts) if nearest_e[nearest_e[j]] == j and j < nearest_e[j])

    return {
        'n_pairs': len(jaccard_p),
        'jaccard_mean': float(jaccard_p.mean()),
        'jaccard_median': float(torch.quantile(jaccard_p, 0.5)),
        'pairs_sharing_no_latent': float((intersection_p == 0).double().mean()),
        'n_top_pairs': n_top,
        'intersection_share_of_top_pairs': float(intersection_p[top_p].sum() / intersection_p.sum()),
        'strong_pair_threshold': _STRONG_PAIR_J,
        'n_strong_pairs': int(strong_p.sum()),
        'strong_pair_fraction': float(strong_p.double().mean()),
        'experts_in_strong_pairs': int((n_partners_e > 0).sum()),
        'experts_with_single_partner': int((n_partners_e == 1).sum()),
        'largest_connected_group': largest_component(linked_ee),
        'jaccard_max': float(jaccard_p.max()),
        'mutual_nearest_neighbour_pairs': n_mutual,
        'strong_pair_overlap_median': float(overlap_p[strong_p].median()) if strong_p.any() else None,
        'private_fraction_mean': float(private_fraction_e.mean()),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / f'tau_{args.tau}'
    summary = {'tau': args.tau, 'layers': {}}

    for layer in args.layers:
        print(f'\nloading layer {layer} ...')
        jaccard_ee, intersection_ee, expert_degree_e, private_fraction_e = overlap_matrices(
            dominance_adjacency(aggregated_path(layer), args.tau)
        )
        stats = overlap_stats(jaccard_ee, intersection_ee, expert_degree_e, private_fraction_e)
        summary['layers'][str(layer)] = stats
        save_safetensors({
            'jaccard': jaccard_ee.float(),
            'intersection': intersection_ee,
            'expert_degree': expert_degree_e,
        }, output_dir / f'jaccard_{layer}.safetensors')

        print(f"=== layer {layer} ===")
        print(f"mean J {stats['jaccard_mean']:.3e}, median J {stats['jaccard_median']:.3e}, "
              f"pairs sharing no latent {stats['pairs_sharing_no_latent'] * 100:.1f}%, "
              f"intersection in the top {stats['n_top_pairs']} pairs "
              f"{stats['intersection_share_of_top_pairs'] * 100:.1f}%")
        print(f"pairs with J > {_STRONG_PAIR_J}: {stats['n_strong_pairs']} "
              f"({stats['strong_pair_fraction'] * 100:.2f}%), experts involved {stats['experts_in_strong_pairs']}, "
              f"with a single partner {stats['experts_with_single_partner']}, "
              f"largest connected group {stats['largest_connected_group']}, largest J {stats['jaccard_max']:.3f}")
        median_overlap = stats['strong_pair_overlap_median']
        median_text = (f"median overlap of pairs above {_STRONG_PAIR_J} {median_overlap * 100:.1f}%"
                       if median_overlap is not None else 'no pair above the threshold')
        print(f"mutual nearest-neighbour pairs {stats['mutual_nearest_neighbour_pairs']}, "
              f"{median_text}, mean private fraction {stats['private_fraction_mean'] * 100:.1f}%")

    save_json(summary, output_dir / 'summary.json')


if __name__ == '__main__':
    main()
