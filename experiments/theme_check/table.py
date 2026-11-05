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
Print Table 2 from a directory of theme_check.py results, named e{expert}_{name}.json for the
Share row and e{expert}_{name}_random.json for the Random row.
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import DATA_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--result_dir', type = str, default = str(DATA_DIR / 'layer30' / 'theme_check'))
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in sorted(Path(args.result_dir).glob('e*.json')):
        if path.stem.endswith('_random'):
            continue
        share = json.loads(path.read_text())
        random_path = path.with_name(f'{path.stem}_random.json')
        random = json.loads(random_path.read_text()) if random_path.exists() else None
        rows.append((share['expert'], share['theme'], share['share'], random and random['share']))

    print(f"{'expert':>7}  {'theme':<28}{'share':>8}{'random':>8}")
    for expert, theme, share, random in sorted(rows):
        print(f'{expert:>7}  {theme:<28}{share * 100:>7.1f}%' + (f'{random * 100:>7.1f}%' if random is not None else ''))


if __name__ == '__main__':
    main()
