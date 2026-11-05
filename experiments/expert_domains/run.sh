#!/bin/bash
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

# Section 5.3: group the expert labels of layer 30 into domains and count the experts per domain.
# Needs OPENAI_API_KEY.
#
# Starts from the released labels in data/layer30/expert_labels.jsonl. Labeling the experts
# (label_experts.py) is a separate step that needs the explanation word clouds of all 128
# experts; to group your own labels instead, run
#
#   python experiments/expert_domains/label_experts.py --output_path outputs/expert_domains/expert_labels.jsonl
#   LABELS=outputs/expert_domains/expert_labels.jsonl bash experiments/expert_domains/run.sh
#
# To reproduce only the statistics from the released labels and groups, run domain_stats.py alone
# without arguments.

set -e
cd "$(dirname "$0")/../.."

LABELS=${LABELS:-data/layer30/expert_labels.jsonl}
OUT=${OUT:-outputs/expert_domains}

python experiments/expert_domains/group_experts.py --label_path $LABELS --output_path $OUT/expert_groups.json
python experiments/expert_domains/domain_stats.py \
  --label_path $LABELS \
  --group_path $OUT/expert_groups.json \
  --output_dir $OUT
