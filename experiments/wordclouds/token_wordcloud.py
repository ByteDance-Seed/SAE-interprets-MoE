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
Section 5.1, Figure 4(a) and 4(b): word clouds of the tokens routed to an expert.

Every routed slot adds its gating weight rather than a count, giving the gating mass m(e, t) that
expert e accumulates on token t. Token ids are decoded one by one and merged into surface forms
after strip() and lower(), so " the", "The" and "the" become one word. Ids that decode to an
incomplete UTF-8 fragment (containing U+FFFD) are dropped.

--weighting count   weight = m(e, t)                                            Figure 4(a)
--weighting mi      weight = m(e, t) * log2(P(t | e) / P(t)), the contribution  Figure 4(b)
                    of the cell to the mutual information I(E; T), with the five
                    most frequent function words removed before taking the top 200

The paper draws expert 72 at layer 30 on the Dolma 3 pretraining part of the validation set,
i.e. on an aggregate built with contribution/aggregate.py --pattern "<pretrain parquet stem>*".

Input: routing_mass of the aggregated contribution statistics, and the tokenizer.
Output: {output_dir}/{weighting}/e{expert}.png and e{expert}.json.

Suffix convention: te all experts, v vocabulary, w merged surface forms.
'''

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import MODEL_NAME, OUTPUT_DIR, aggregated_path
from experiments.wordclouds.explanation_wordcloud import DEFAULT_FONT, HEIGHT, MAX_WORDS, SEED, WIDTH

_REPLACEMENT_CHAR = '\ufffd'
_FILTERED_WORDS = ('the', 'of', 'to', 'and', 'a')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--aggregated_path', type = str, default = str(aggregated_path(30)))
    p.add_argument('--model_path', type = str, default = MODEL_NAME)
    p.add_argument('--weighting', type = str, required = True, choices = ['count', 'mi'])
    p.add_argument('--experts', type = int, nargs = '+', default = None, help = 'default: every expert')
    p.add_argument('--font_path', type = str, default = DEFAULT_FONT if os.path.exists(DEFAULT_FONT) else None)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'wordclouds' / 'token'))
    return p.parse_args()


def surface_forms(tokenizer, vocab_size):
    decoded = tokenizer.batch_decode([[token_id] for token_id in range(vocab_size)])
    form_to_index = {}
    group_v = torch.full((vocab_size,), -1, dtype = torch.int64)
    for token_id, text in enumerate(decoded):
        surface = text.strip().lower()
        if not surface or _REPLACEMENT_CHAR in surface:
            continue
        group_v[token_id] = form_to_index.setdefault(surface, len(form_to_index))
    return list(form_to_index), group_v


def main():
    args = parse_args()

    with safe_open(args.aggregated_path, framework = 'pt') as f:
        mass_texv = f.get_tensor('routing_mass')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    forms, group_v = surface_forms(tokenizer, mass_texv.shape[1])

    kept_v = group_v >= 0
    mass_texw = torch.stack([
        torch.bincount(group_v[kept_v], weights = mass_v[kept_v], minlength = len(forms))
        for mass_v in mass_texv
    ])
    print(f'{int(kept_v.sum())} token ids -> {len(forms)} surface forms, '
          f'kept {float(mass_texw.sum() / mass_texv.sum()) * 100:.2f}% of the routing mass.')

    if args.weighting == 'count':
        weight_texw = mass_texw
    else:
        p_w = mass_texw.sum(dim = 0) / mass_texw.sum()
        p_texw = mass_texw / mass_texw.sum(dim = 1, keepdim = True)
        # xlogy gives 0 * log 0 = 0 for surface forms an expert never receives
        weight_texw = (torch.xlogy(mass_texw, p_texw) - torch.xlogy(mass_texw, p_w)) / math.log(2)
        weight_texw[:, [forms.index(word) for word in _FILTERED_WORDS]] = -torch.inf

    output_dir = Path(args.output_dir) / args.weighting
    output_dir.mkdir(parents = True, exist_ok = True)
    for expert_id in args.experts or range(mass_texw.shape[0]):
        top = torch.topk(weight_texw[expert_id], k = MAX_WORDS)
        words = [{'word': forms[index], 'weight': weight} for index, weight in zip(top.indices.tolist(), top.values.tolist())]
        with open(output_dir / f'e{expert_id}.json', 'w') as f:
            json.dump(words, f, ensure_ascii = False, indent = 2)
        WordCloud(
            width = WIDTH,
            height = HEIGHT,
            max_words = MAX_WORDS,
            font_path = args.font_path,
            background_color = 'white',
            colormap = 'viridis',
            prefer_horizontal = 0.9,
            random_state = SEED,
        ).generate_from_frequencies({record['word']: record['weight'] for record in words}).to_file(output_dir / f'e{expert_id}.png')

    print(f'write finished to {output_dir}.')


if __name__ == '__main__':
    main()
