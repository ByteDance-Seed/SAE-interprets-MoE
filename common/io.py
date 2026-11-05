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
import os

from safetensors import safe_open
from safetensors.torch import save_file

_EPS = 1e-9


def save_safetensors(tensors, dest_path):
    '''Write to a temporary file first so that an interrupted job never leaves a truncated file behind.'''
    dest_path = str(dest_path)
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok = True)
    tmp_path = f'{dest_path}.tmp'
    save_file({key: value.contiguous().cpu() for key, value in tensors.items()}, tmp_path)
    os.replace(tmp_path, dest_path)
    print(f'write finished to {dest_path}.')


def save_json(obj, dest_path):
    dest_path = str(dest_path)
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok = True)
    tmp_path = f'{dest_path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, ensure_ascii = False, indent = 2)
    os.replace(tmp_path, dest_path)
    print(f'write finished to {dest_path}.')


def load_share(aggregated_path):
    '''Contribution fraction s(j, i) of Eq. (9), shape [n_experts, n_latents].'''
    with safe_open(str(aggregated_path), framework = 'pt') as f:
        C_exf = f.get_tensor('C')
    return C_exf / C_exf.sum(dim = 0, keepdim = True).clamp_min(_EPS)


def dominance_adjacency(aggregated_path, tau):
    '''Edges of the dominance graph: expert j dominates latent i iff s(j, i) >= tau.'''
    return load_share(aggregated_path) >= tau
