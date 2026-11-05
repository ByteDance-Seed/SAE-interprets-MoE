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
Section 5.3, Figure 5 and Appendix table of expert labels: how the experts of layer 30 distribute
over the recovered domains, and whether the most overlapping expert pairs fall inside a domain.

Reported:
- the number and share of experts in every domain, largest first;
- given the Jaccard matrix of experiments/dominance_overlap at layer 30, how many of the --n_pairs
  pairs with the largest J join two experts of the same domain, against the share of all expert
  pairs that do, and the median share of an expert's dominant set taken up by the latents it
  shares with its partner, over both experts of every such pair.

The bar chart and the label table are also written as LaTeX snippets (pgfplots / longtable with
multirow and array) that the paper \\input's.

By default the labels and groups released with the paper are used, so this runs without API access.

Output: {output_dir}/domain_stats.json, expert_group_share_plot.tex, expert_labels_table.tex
'''

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import save_json
from common.paths import DATA_DIR, OUTPUT_DIR
from experiments.expert_domains.label_experts import read_labels

_LATEX_SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_'}
_PLOT_COLOURS = ('blue', 'red', 'teal', 'orange', 'violet', 'olive', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'purple')

_PLOT_HEAD = r'''\begin{tikzpicture}
  \begin{axis}[
    width = 0.86\textwidth, height = 0.25\textwidth,
    ybar,
    area legend,
    bar width = 16pt,
    xlabel = {domain, ordered by the number of experts it holds},
    ylabel = {number of experts},
    xmin = 0.4, xmax = {xmax},
    ymin = 0, ymax = {ymax},
    xtick = \empty,
    grid = major, grid style = {gray!20},
    tick align = outside,
    nodes near coords,
    every node near coord/.append style = {font = \scriptsize, inner sep = 2pt},
    legend columns = 3,
    legend cell align = left,
    legend style = {
      at = {(0.5,-0.32)}, anchor = north, draw = none, font = \scriptsize,
      /tikz/every even column/.append style = {column sep = 1.5em},
    },
  ]
'''

_TABLE_HEAD = r'''\small\begin{longtable}{>{\raggedright\arraybackslash}p{2cm}
                  p{1cm}
                  >{\raggedright\arraybackslash}p{\dimexpr\linewidth-3cm-6\tabcolsep\relax}}
  \caption{The {n_experts} experts of layer 30, their labels and the domains they were grouped
  into. A label is written by GPT-5.5 from the explanation word cloud of that expert alone, and
  the domains are recovered from the labels in a single further call. Both prompts are given in
  \Cref{app:expert-label-prompt} and \Cref{app:expert-group-prompt}.}
  \label{tab:expert-labels} \\
  \hline
  Domain & Expert & Label \\
  \hline
  \endfirsthead

  \hline
  Domain & Expert & Label \\
  \hline
  \endhead

  \hline
  \endfoot
'''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--label_path', type = str, default = str(DATA_DIR / 'layer30' / 'expert_labels.jsonl'))
    p.add_argument('--group_path', type = str, default = str(DATA_DIR / 'layer30' / 'expert_groups.json'))
    p.add_argument('--jaccard_path', type = str,
                   default = str(OUTPUT_DIR / 'dominance_overlap' / 'tau_0.2' / 'jaccard_30.safetensors'))
    p.add_argument('--n_pairs', type = int, default = 20)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'expert_domains'))
    return p.parse_args()


def latex_escape(text):
    return ''.join(_LATEX_SPECIAL.get(char, char) for char in text)


def write_plot(groups, dest_path):
    assert len(groups) <= len(_PLOT_COLOURS), f'only {len(_PLOT_COLOURS)} colours in the palette'
    counts = [len(group['experts']) for group in groups]
    lines = [_PLOT_HEAD.replace('{xmax}', f'{len(groups) + 0.6:.1f}').replace('{ymax}', f'{max(counts) * 1.16:.0f}')]
    for g, (group, count) in enumerate(zip(groups, counts)):
        colour = _PLOT_COLOURS[g]
        lines.append(f'    \\addplot[fill = {colour}!30, draw = {colour}!70!black, bar shift = 0pt] '
                     f'coordinates {{({g + 1},{count})}};\n')
        lines.append(f"    \\addlegendentry{{{latex_escape(group['group'])}}}\n")
    lines.append('  \\end{axis}\n\\end{tikzpicture}\n')
    Path(dest_path).write_text(''.join(lines))
    print(f'write finished to {dest_path}.')


def write_table(groups, labels, dest_path):
    lines = [_TABLE_HEAD.replace('{n_experts}', str(len(labels)))]
    for g, group in enumerate(groups):
        if g:
            lines.append('  \\hline\n')
        for e, expert_id in enumerate(group['experts']):
            domain = f"\\multirow[t]{{{len(group['experts'])}}}{{=}}{{\\raggedright {latex_escape(group['group'])}}}" if e == 0 else ''
            lines.append(f'  {domain} & {expert_id} & {latex_escape(labels[expert_id])} \\\\\n')
    lines.append('  \\hline\n\\end{longtable}\n')
    Path(dest_path).write_text(''.join(lines))
    print(f'write finished to {dest_path}.')


def overlap_vs_domain(jaccard_path, domain_of, n_pairs):
    with safe_open(jaccard_path, framework = 'pt') as f:
        jaccard_ee = f.get_tensor('jaccard').double()
        intersection_ee = f.get_tensor('intersection').double()
        expert_degree_e = f.get_tensor('expert_degree').double()
    n_experts = jaccard_ee.shape[0]
    row_p, col_p = torch.triu_indices(n_experts, n_experts, offset = 1)
    top_p = jaccard_ee[row_p, col_p].argsort(descending = True)[:n_pairs]

    top_pairs = []
    shared_fraction = []
    for p in top_p.tolist():
        a, b = int(row_p[p]), int(col_p[p])
        top_pairs.append({'experts': [a, b], 'jaccard': float(jaccard_ee[a, b]),
                          'same_domain': domain_of[a] == domain_of[b]})
        shared_fraction += [float(intersection_ee[a, b] / expert_degree_e[a]),
                            float(intersection_ee[a, b] / expert_degree_e[b])]
    same_all = [domain_of[int(a)] == domain_of[int(b)] for a, b in zip(row_p, col_p)]
    return {
        'top_pairs': top_pairs,
        'top_pairs_in_same_domain': sum(pair['same_domain'] for pair in top_pairs),
        'all_pairs_in_same_domain_fraction': sum(same_all) / len(same_all),
        'shared_fraction_of_dominant_set_median': float(torch.tensor(shared_fraction).median()),
    }


def main():
    args = parse_args()
    labels = read_labels(args.label_path)
    groups = sorted(json.loads(Path(args.group_path).read_text()), key = lambda group: len(group['experts']), reverse = True)
    domain_of = {expert_id: group['group'] for group in groups for expert_id in group['experts']}
    assert len(domain_of) == sum(len(group['experts']) for group in groups), 'an expert appears in more than one group'
    assert set(domain_of) == set(labels), 'the grouping is not a partition of the labeled experts'
    n_experts = len(labels)

    result = {
        'n_experts': n_experts,
        'n_domains': len(groups),
        'domains': [{
            'domain': group['group'],
            'n_experts': len(group['experts']),
            'share': len(group['experts']) / n_experts,
            'experts': [{'expert': expert_id, 'label': labels[expert_id]} for expert_id in group['experts']],
        } for group in groups],
    }
    print(f"{'domain':>28}{'experts':>9}{'share':>9}")
    for domain in result['domains']:
        print(f"{domain['domain']:>28}{domain['n_experts']:>9}{domain['share']:>9.1%}")

    if Path(args.jaccard_path).exists():
        result['overlap_vs_domain'] = overlap_vs_domain(args.jaccard_path, domain_of, args.n_pairs)
        stats = result['overlap_vs_domain']
        print(f"\nof the {args.n_pairs} pairs with the largest J, {stats['top_pairs_in_same_domain']} join two experts "
              f"of the same domain, against {stats['all_pairs_in_same_domain_fraction'] * 100:.1f}% of all pairs; "
              f"shared latents take a median {stats['shared_fraction_of_dominant_set_median'] * 100:.1f}% "
              f"of the dominant sets of these experts")
    else:
        print(f'\n{args.jaccard_path} not found, run experiments/dominance_overlap first for the pair statistics.')

    output_dir = Path(args.output_dir)
    save_json(result, output_dir / 'domain_stats.json')
    write_plot(groups, output_dir / 'expert_group_share_plot.tex')
    write_table(groups, labels, output_dir / 'expert_labels_table.tex')


if __name__ == '__main__':
    main()
