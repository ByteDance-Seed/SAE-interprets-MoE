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

import json
import re
from pathlib import Path

from .paths import DATA_DIR

EXPLANATION_DIR = DATA_DIR / 'layer30' / 'explanations'
UNINTERPRETABLE = 'Uninterpretable'

_FILE_PATTERN = re.compile(r'^e(\d+)\.jsonl$')


def find_expert_ids(explanation_dir = EXPLANATION_DIR):
    expert_ids = sorted(
        int(matched.group(1))
        for path in Path(explanation_dir).iterdir()
        if (matched := _FILE_PATTERN.match(path.name))
    )
    assert expert_ids, f'no explanation file matching e*.jsonl in {explanation_dir}'
    return expert_ids


def read_explanations(expert_id, explanation_dir = EXPLANATION_DIR):
    '''
    Interpretable explanations of the latents sampled from one expert's dominant set, as a list of
    {'feat', 'explanation'} in file order. Latents whose explanation is missing or exactly
    "Uninterpretable" do not describe the expert and are dropped.
    '''
    records = []
    with open(Path(explanation_dir) / f'e{expert_id}.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            explanation = record['explanation']
            if explanation is None or explanation == UNINTERPRETABLE:
                continue
            records.append({'feat': record['feat'], 'explanation': explanation})
    assert records, f'no interpretable latent for expert {expert_id} in {explanation_dir}'
    return records
