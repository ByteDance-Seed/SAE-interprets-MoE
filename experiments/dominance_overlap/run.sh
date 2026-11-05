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

# Section 4.3, Table 1 and the appendix Jaccard heat map. Needs the aggregated statistics of
# contribution/run.sh.

set -e
cd "$(dirname "$0")/../.."

python experiments/dominance_overlap/overlap.py --layers 4 30 46 --tau 0.2
python experiments/dominance_overlap/plot_heatmap.py --layers 4 30 46 --overlap_dir outputs/dominance_overlap/tau_0.2
