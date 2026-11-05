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
Appendix D, Figure 8: heatmaps of the pairwise Jaccard similarity of the dominant sets.

The three layers share one linear colour scale saturating at 0.05, the experts stay in their
original index order on both axes, and the diagonal is masked.

Input: the jaccard_{layer}.safetensors written by overlap.py.
Output: {overlap_dir}/jaccard_heatmap.pdf (and .png).
'''

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import LAYERS, OUTPUT_DIR

_VMAX = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type = int, nargs = '+', default = list(LAYERS))
    p.add_argument('--overlap_dir', type = str, default = str(OUTPUT_DIR / 'dominance_overlap' / 'tau_0.2'))
    return p.parse_args()


def main():
    args = parse_args()
    n_layers = len(args.layers)
    fig, axes = plt.subplots(1, n_layers, figsize = (4.1 * n_layers, 4.2), constrained_layout = True)
    axes = np.atleast_1d(axes)
    colour = matplotlib.colormaps['magma'].with_extremes(bad = matplotlib.colormaps['magma'](0.0))

    for ax, layer in zip(axes, args.layers):
        with safe_open(f'{args.overlap_dir}/jaccard_{layer}.safetensors', framework = 'pt') as f:
            jaccard_ee = f.get_tensor('jaccard').double().numpy()
        np.fill_diagonal(jaccard_ee, np.nan)
        image = ax.imshow(jaccard_ee, cmap = colour, interpolation = 'nearest', vmin = 0.0, vmax = _VMAX)
        ax.set_title(f'layer {layer}')
        ax.set_xlabel('expert id')
        ax.set_ylabel('expert id' if ax is axes[0] else '')

    bar = fig.colorbar(image, ax = axes, fraction = 0.02, pad = 0.02, extend = 'max')
    bar.set_label('Jaccard similarity')
    for extension in ('pdf', 'png'):
        dest_path = f'{args.overlap_dir}/jaccard_heatmap.{extension}'
        fig.savefig(dest_path, dpi = 160)
        print(f'write finished to {dest_path}.')


if __name__ == '__main__':
    main()
