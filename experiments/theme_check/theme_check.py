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
Section 5.2 and Table 2: share of an expert's latent explanations that belong to a theme.

Default mode judges every interpretable explanation of --expert against --theme with one
yes / no request each, giving the Share row. --random draws --sample_size explanations uniformly
from the pooled interpretable explanations of all other experts and judges those instead, giving
the Random row. The pool is built in expert order and file order, so a given seed always draws the
same sample. --random needs the explanations of all experts of the layer and refuses to run with
fewer.

Results are written after every run and existing judgements are reused, so rerunning the same
command retries only the requests that failed.

The judgements behind Table 2 are released in data/layer30/theme_check, and table.py prints the
table from a directory of results.

Input: data/layer30/explanations/e{expert}.jsonl and prompt.txt
Output: --output_path, a json with the share and every judgement.
'''

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.explanations import EXPLANATION_DIR, find_expert_ids, read_explanations
from common.llm import Chat, add_llm_args
from common.paths import N_EXPERTS

_PROMPT_PATH = Path(__file__).with_name('prompt.txt')
_YES, _NO = {'yes', 'y'}, {'no', 'n'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--theme', type = str, required = True, help = 'e.g. "Java development"')
    p.add_argument('--expert', type = int, required = True)
    p.add_argument('--random', action = 'store_true', help = 'sample from all other experts instead')
    p.add_argument('--sample_size', type = int, default = 2000)
    p.add_argument('--seed', type = int, default = 0)
    p.add_argument('--explanation_dir', type = str, default = str(EXPLANATION_DIR))
    p.add_argument('--output_path', type = str, required = True)
    add_llm_args(p, max_tokens = 4000)
    return p.parse_args()


def collect_candidates(args):
    if not args.random:
        return [{**record, 'expert': args.expert} for record in read_explanations(args.expert, args.explanation_dir)]
    expert_ids = find_expert_ids(args.explanation_dir)
    if len(expert_ids) < N_EXPERTS:
        sys.exit(f'--random draws from all other experts and needs the explanations of all {N_EXPERTS} experts, '
                 f'found {len(expert_ids)} in {args.explanation_dir}. The random judgements behind Table 2 '
                 'are released in data/layer30/theme_check.')
    pool = [
        {**record, 'expert': expert_id}
        for expert_id in expert_ids if expert_id != args.expert
        for record in read_explanations(expert_id, args.explanation_dir)
    ]
    return random.Random(args.seed).sample(pool, min(args.sample_size, len(pool)))


def parse_yes_no(content):
    answer = content.strip().lower().rstrip('.')
    first = answer.split(None, 1)[0].rstrip('.,') if answer else ''
    for word in (answer, first):
        if word in _YES:
            return True
        if word in _NO:
            return False
    raise ValueError(f'expected yes/no, got {content!r}')


def record_key(record):
    # a latent can be dominated by several experts and so appear in the pool more than once
    return record['expert'], record['feat']


def write_results(args, judged, candidates):
    results = [{**record, 'belongs': judged[record_key(record)]} for record in candidates if record_key(record) in judged]
    n_positive = sum(record['belongs'] for record in results)
    payload = {
        'theme': args.theme,
        'expert': args.expert,
        'random': args.random,
        'seed': args.seed if args.random else None,
        'n_checked': len(results),
        'n_positive': n_positive,
        'share': n_positive / len(results) if results else None,
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok = True)
    tmp_path = f'{args.output_path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, ensure_ascii = False, indent = 2)
    os.replace(tmp_path, args.output_path)
    return payload


def main():
    args = parse_args()
    candidates = collect_candidates(args)

    judged = {}
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            judged = {record_key(record): record['belongs'] for record in json.load(f)['results']}
    pending = [record for record in candidates if record_key(record) not in judged]
    print(f"{'random' if args.random else 'expert'} mode, theme {args.theme!r}, expert {args.expert}: "
          f'{len(candidates)} candidates, {len(candidates) - len(pending)} already judged, {len(pending)} to go')

    prompt_prefix = _PROMPT_PATH.read_text().replace('{theme}', args.theme)
    for record, belongs in Chat(args).map(lambda chat, item: parse_yes_no(chat(prompt_prefix + item['explanation'])), pending):
        judged[record_key(record)] = belongs

    payload = write_results(args, judged, candidates)
    print(f"share {payload['share']:.1%} ({payload['n_positive']}/{payload['n_checked']}), "
          f'write finished to {args.output_path}.')


if __name__ == '__main__':
    main()
