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

# Section 5.1, Figures 3 and 4.
#
# The explanation word clouds are drawn for every expert found in data/layer30/explanations.
#
# The token word clouds of expert 72 need the routing mass of contribution/run.sh, aggregated over
# the Dolma 3 pretraining part of the validation set only. Set PRETRAIN_PATTERN to the glob that
# selects those parts (e.g. PRETRAIN_PATTERN="dolma3_*"); without it the token clouds are skipped.

set -e
cd "$(dirname "$0")/../.."

python experiments/wordclouds/explanation_wordcloud.py

if [ -n "$PRETRAIN_PATTERN" ]; then
  PRETRAIN_AGGREGATE=outputs/contribution/layer30/aggregated_pretrain.safetensors
  python contribution/aggregate.py --layer 30 --pattern "$PRETRAIN_PATTERN" --output $PRETRAIN_AGGREGATE
  for weighting in count mi; do
    python experiments/wordclouds/token_wordcloud.py --aggregated_path $PRETRAIN_AGGREGATE \
      --weighting $weighting --experts 72
  done
fi
