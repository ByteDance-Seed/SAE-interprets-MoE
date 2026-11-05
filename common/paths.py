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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

MODEL_NAME = 'Qwen/Qwen3-30B-A3B'
LAYERS = (4, 30, 46)
N_EXPERTS = 128
TOP_K = 8


def contribution_dir(layer):
    return OUTPUT_DIR / 'contribution' / f'layer{layer}'


def aggregated_path(layer):
    return contribution_dir(layer) / 'aggregated.safetensors'
