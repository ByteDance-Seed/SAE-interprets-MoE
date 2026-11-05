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

# Table 2: theme shares of experts 50, 57, 1 and 76 at layer 30. Needs OPENAI_API_KEY. Every
# command can be rerun to retry failed requests.
#
# The Random row draws from the explanations of all 128 experts, so it is only recomputed with
# WITH_RANDOM=1 and a complete explanation directory. The released random judgements are in
# data/layer30/theme_check; copy them into $OUT to print the full table.

set -e
cd "$(dirname "$0")/../.."

OUT=${OUT:-outputs/theme_check}

run() {
  local name=$1 theme=$2 expert=$3
  python experiments/theme_check/theme_check.py --theme "$theme" --expert $expert \
    --output_path $OUT/e${expert}_${name}.json
  if [ -n "$WITH_RANDOM" ]; then
    python experiments/theme_check/theme_check.py --theme "$theme" --expert $expert --random \
      --output_path $OUT/e${expert}_${name}_random.json
  fi
}

run java                   "Java development"          50
run frontend               "Front-end development"     57
run geometry               "Geometry"                  1
run arithmetic_polynomial  "Arithmetic and polynomial" 76

python experiments/theme_check/table.py --result_dir $OUT
