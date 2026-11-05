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
Section 5.1, Figures 3 and 4(c), Appendix word clouds: one explanation word cloud per expert.

All interpretable explanations of an expert are joined into one text and handed to
WordCloud.generate unchanged, which tokenizes, removes stopwords, merges frequent two-word
collocations and plurals, and weights a word by its frequency relative to the most frequent one.
Besides the default stopword list only "say" is removed, the prefix of output-feature
explanations ("Say X"), which recurs in every expert without describing any of them.

The word weights are saved alongside the image, the expert_domains experiment labels each expert
from them.

Input: data/layer30/explanations/e{expert}.jsonl
Output: {output_dir}/e{expert}.png and e{expert}.json ([{"word", "weight"}], heaviest first).
'''

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from wordcloud import STOPWORDS, WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.explanations import EXPLANATION_DIR, find_expert_ids, read_explanations
from common.paths import OUTPUT_DIR

# needs CJK coverage, otherwise Chinese and Japanese words render as boxes
DEFAULT_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
WIDTH, HEIGHT = 1600, 900
MAX_WORDS = 200
SEED = 721

_EXTRA_STOPWORDS = {'say'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--explanation_dir', type = str, default = str(EXPLANATION_DIR))
    p.add_argument('--experts', type = int, nargs = '+', default = None, help = 'default: every expert found')
    p.add_argument('--font_path', type = str, default = DEFAULT_FONT if os.path.exists(DEFAULT_FONT) else None)
    p.add_argument('--output_dir', type = str, default = str(OUTPUT_DIR / 'wordclouds' / 'explanation'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok = True)
    expert_ids = args.experts or find_expert_ids(args.explanation_dir)

    for expert_id in tqdm(expert_ids):
        explanations = [record['explanation'] for record in read_explanations(expert_id, args.explanation_dir)]
        word_cloud = WordCloud(
            width = WIDTH,
            height = HEIGHT,
            max_words = MAX_WORDS,
            font_path = args.font_path,
            stopwords = STOPWORDS | _EXTRA_STOPWORDS,
            background_color = 'white',
            colormap = 'viridis',
            prefer_horizontal = 0.9,
            random_state = SEED,
        ).generate('\n'.join(explanations))
        word_cloud.to_file(f'{args.output_dir}/e{expert_id}.png')
        with open(f'{args.output_dir}/e{expert_id}.json', 'w') as f:
            json.dump([{'word': word, 'weight': weight} for word, weight in word_cloud.words_.items()],
                      f, ensure_ascii = False, indent = 2)

    print(f'write finished to {args.output_dir}.')


if __name__ == '__main__':
    main()
