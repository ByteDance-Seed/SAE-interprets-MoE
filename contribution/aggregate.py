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
Sum the partial results written by contribution/trace.py for one layer. Every tensor in a part is
a sum over its tokens, so the aggregate is a plain sum, and the contribution fraction of Eq. (9)
follows from C by normalizing over experts (common.io.load_share).

--pattern selects a subset of the parts, e.g. "dolma3_*" to aggregate a single data source.

Output: {output_dir}/layer{layer}/aggregated.safetensors unless --output is given.
'''

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import save_safetensors
from common.paths import OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type = int, required = True)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'contribution'))
    p.add_argument('--pattern', type = str, default = '*')
    p.add_argument('--output', type = str, default = None)
    return p.parse_args()


def main():
    args = parse_args()
    layer_dir = Path(args.output_dir) / f'layer{args.layer}'
    part_files = sorted((layer_dir / 'parts').glob(f'{args.pattern}.safetensors'))
    assert part_files, f'no part matching {args.pattern}.safetensors under {layer_dir / "parts"}'
    print(f'aggregating {len(part_files)} parts.')

    totals = None
    for part_file in tqdm(part_files):
        with safe_open(str(part_file), framework = 'pt') as f:
            if totals is None:
                totals = {key: f.get_tensor(key) for key in f.keys()}
            else:
                for key in totals:
                    totals[key] += f.get_tensor(key)

    act_count_f = totals['act_count']
    print(f"{int(totals['n_tokens'])} tokens, {int(totals['n_pairs'])} (token, active latent) pairs, "
          f"{int((act_count_f == 0).sum())} of {act_count_f.numel()} latents never active.")
    save_safetensors(totals, args.output or layer_dir / 'aggregated.safetensors')


if __name__ == '__main__':
    main()
