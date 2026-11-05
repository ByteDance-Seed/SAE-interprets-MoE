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

# Contribution statistics shared by every experiment.
#
#   bash contribution/run.sh /path/to/validation/*.parquet
#
# Each parquet file is traced in one call and the parts are then summed per layer. To spread the
# work over many GPUs, launch trace.py yourself on disjoint row ranges (--idx_start / --idx_end)
# and run aggregate.py once all of them have finished; parts are named after file and row range.
#
# Qwen3-30B-A3B in bf16 takes about 61 GB and each SAE encoder another 8 GB in fp32, so hooking
# all three layers in one pass needs two 80 GB GPUs (device_map="auto" spreads the model).
# With a single GPU run one layer at a time, e.g. LAYERS="30" bash contribution/run.sh ...

set -e
cd "$(dirname "$0")/.."

LAYERS=${LAYERS:-"4 30 46"}

for parquet in "$@"; do
  python contribution/trace.py --layers $LAYERS --parquet_path "$parquet"
done

for layer in $LAYERS; do
  python contribution/aggregate.py --layer $layer
done
