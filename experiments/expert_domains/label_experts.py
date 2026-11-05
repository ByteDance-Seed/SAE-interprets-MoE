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
Section 5.3, first step: name every expert from its explanation word cloud.

The heaviest --top_k words of an expert's word cloud are appended to label_prompt.txt as one
"word: weight" line each, in decreasing weight, and the model answers with a label of at most
nine words. One request per expert.

Existing labels are kept and only missing ones are requested, so rerunning retries failures. The
labels used in the paper are released in data/layer30/expert_labels.jsonl; this script never
writes there.

Input: the e{expert}.json word weights written by experiments/wordclouds/explanation_wordcloud.py,
one per expert of the layer
Output: --output_path, one {"expert": id, "label": "..."} per line in expert order.
'''

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import Chat, add_llm_args
from common.paths import OUTPUT_DIR

_PROMPT_PATH = Path(__file__).with_name('label_prompt.txt')
_FILE_PATTERN = re.compile(r'^e(\d+)\.json$')
_MAX_LABEL_WORDS = 9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wordcloud_dir', type = str, default = str(OUTPUT_DIR / 'wordclouds' / 'explanation'))
    p.add_argument('--output_path', type = str, default = str(OUTPUT_DIR / 'expert_domains' / 'expert_labels.jsonl'))
    p.add_argument('--top_k', type = int, default = 200, help = 'number of heaviest words fed into the prompt')
    add_llm_args(p, max_tokens = 4000)
    return p.parse_args()


def read_wordclouds(wordcloud_dir, top_k):
    wordclouds = {}
    for path in Path(wordcloud_dir).iterdir():
        if matched := _FILE_PATTERN.match(path.name):
            entries = sorted(json.loads(path.read_text()), key = lambda entry: entry['weight'], reverse = True)[:top_k]
            wordclouds[int(matched.group(1))] = '\n'.join(f"{entry['word']}: {entry['weight']:.3f}" for entry in entries)
    assert wordclouds, f'no e*.json word cloud in {wordcloud_dir}'
    return wordclouds


def read_labels(label_path):
    if not os.path.exists(label_path):
        return {}
    with open(label_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {record['expert']: record['label'] for record in records if record['label'].strip()}


def write_labels(labels, label_path):
    os.makedirs(os.path.dirname(label_path) or '.', exist_ok = True)
    tmp_path = f'{label_path}.tmp'
    with open(tmp_path, 'w') as f:
        for expert_id in sorted(labels):
            f.write(json.dumps({'expert': expert_id, 'label': labels[expert_id]}, ensure_ascii = False) + '\n')
    os.replace(tmp_path, label_path)
    print(f'{len(labels)} experts labeled, write finished to {label_path}.')


def main():
    args = parse_args()
    wordclouds = read_wordclouds(args.wordcloud_dir, args.top_k)
    labels = read_labels(args.output_path)
    pending = [(expert_id, text) for expert_id, text in sorted(wordclouds.items()) if expert_id not in labels]
    print(f'{len(wordclouds)} word clouds, {len(labels)} experts already labeled, {len(pending)} to go')

    prompt_prefix = _PROMPT_PATH.read_text()
    for (expert_id, _), label in Chat(args).map(lambda chat, item: chat(prompt_prefix + item[1]), pending):
        if len(label.split()) > _MAX_LABEL_WORDS:
            print(f'[warn] expert {expert_id} got a label longer than {_MAX_LABEL_WORDS} words: {label}')
        labels[expert_id] = label
    write_labels(labels, args.output_path)


if __name__ == '__main__':
    main()
