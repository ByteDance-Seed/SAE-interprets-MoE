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
Section 5.3, second step: partition the experts into coarse domains from their labels alone.

All labels go into a single request, as "E{expert}: {label}" lines appended to group_prompt.txt,
because asking expert by expert would either require domain names fixed in advance or produce
names that never add up to a partition. The prompt only bounds the number of groups (6 to 13,
aiming for 10) and names no domain.

A response is accepted only if it is an exact partition of the labeled experts with a group count
inside the bounds; otherwise the request is repeated up to --n_retry times.

Input: the released expert labels in data/layer30/expert_labels.jsonl by default, or labels
written by label_experts.py
Output: --output_path, [{"group": name, "experts": [ids]}] from the largest group to the smallest.
'''

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.io import save_json
from common.llm import Chat, add_llm_args
from common.paths import DATA_DIR, OUTPUT_DIR
from experiments.expert_domains.label_experts import read_labels

_PROMPT_PATH = Path(__file__).with_name('group_prompt.txt')
_FENCE_PATTERN = re.compile(r'^```[a-zA-Z]*\n|\n```$')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--label_path', type = str, default = str(DATA_DIR / 'layer30' / 'expert_labels.jsonl'))
    p.add_argument('--output_path', type = str, default = str(OUTPUT_DIR / 'expert_domains' / 'expert_groups.json'))
    p.add_argument('--min_groups', type = int, default = 6)
    p.add_argument('--max_groups', type = int, default = 13)
    p.add_argument('--target_groups', type = int, default = 10)
    p.add_argument('--n_retry', type = int, default = 3)
    # the reasoning over 128 labels plus a full partition needs a much larger budget than a single label
    add_llm_args(p, max_tokens = 32000)
    args = p.parse_args()
    if not 0 < args.min_groups <= args.target_groups <= args.max_groups:
        p.error('need 0 < min_groups <= target_groups <= max_groups')
    return args


def build_prompt(labels, args):
    prompt = _PROMPT_PATH.read_text()
    # the prompt contains literal braces in its JSON example, so no str.format
    for placeholder, value in (('{n_experts}', len(labels)), ('{min_groups}', args.min_groups),
                               ('{max_groups}', args.max_groups), ('{target_groups}', args.target_groups)):
        prompt = prompt.replace(placeholder, str(value))
    return prompt + '\n'.join(f'E{expert_id}: {labels[expert_id]}' for expert_id in sorted(labels))


def parse_groups(content, expert_ids, args):
    groups = json.loads(_FENCE_PATTERN.sub('', content.strip()).strip())
    assert isinstance(groups, list), f'expected a list, got {type(groups).__name__}'
    assert args.min_groups <= len(groups) <= args.max_groups, \
        f'got {len(groups)} groups, expected between {args.min_groups} and {args.max_groups}'
    groups = [{'group': group['group'].strip(), 'experts': sorted(int(e) for e in group['experts'])} for group in groups]
    assigned = [expert_id for group in groups for expert_id in group['experts']]
    assert all(group['group'] for group in groups), 'empty group name'
    assert all(group['experts'] for group in groups), 'empty group'
    assert len(assigned) == len(set(assigned)), 'an expert is assigned to more than one group'
    assert set(assigned) == set(expert_ids), \
        f'missing experts {sorted(set(expert_ids) - set(assigned))}, unknown experts {sorted(set(assigned) - set(expert_ids))}'
    return sorted(groups, key = lambda group: len(group['experts']), reverse = True)


def main():
    args = parse_args()
    labels = read_labels(args.label_path)
    assert labels, f'no label in {args.label_path}'
    prompt = build_prompt(labels, args)
    chat = Chat(args)

    groups = None
    for attempt in range(args.n_retry + 1):
        try:
            groups = parse_groups(chat(prompt), labels.keys(), args)
            break
        except Exception as e:
            print(f'[warn] attempt {attempt + 1}/{args.n_retry + 1} rejected: {type(e).__name__}: {e}')
    assert groups is not None, f'no valid partition after {args.n_retry + 1} attempts'

    print(f'{len(groups)} groups over {len(labels)} experts:')
    for group in groups:
        print(f"  {len(group['experts']):3d}  {group['group']}")
    save_json(groups, args.output_path)


if __name__ == '__main__':
    main()
