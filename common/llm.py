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
Thin wrapper around an OpenAI-compatible chat completion endpoint. The paper uses GPT-5.5 for
judging theme membership, labeling experts and grouping them into domains.

Set OPENAI_API_KEY, and OPENAI_BASE_URL if you go through a proxy or another compatible provider.
'''

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = 'gpt-5.5-2026-04-24'


def add_llm_args(p, max_tokens):
    p.add_argument('--model', type = str, default = DEFAULT_MODEL)
    p.add_argument('--max_tokens', type = int, default = max_tokens,
                   help = 'completion budget, including the reasoning tokens of reasoning models')
    p.add_argument('--qps', type = float, default = 4.0, help = 'max requests per second, <= 0 means unlimited')
    p.add_argument('--workers', type = int, default = 16, help = 'number of concurrent requests in flight')


class RateLimiter:
    def __init__(self, qps):
        self.min_interval = 1.0 / qps if qps > 0 else 0.0
        self.last_request = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return
        # sleep while holding the lock, so that request start times are at least min_interval apart
        with self.lock:
            elapsed = time.monotonic() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.monotonic()


class Chat:
    def __init__(self, args):
        self.client = OpenAI()
        self.model = args.model
        self.max_tokens = args.max_tokens
        self.limiter = RateLimiter(args.qps)
        self.workers = args.workers

    def __call__(self, prompt):
        self.limiter.wait()
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [{'role': 'user', 'content': prompt}],
            max_completion_tokens = self.max_tokens,
        )
        choice = response.choices[0]
        content = (choice.message.content or '').strip()
        # a reasoning model that runs out of budget returns an empty body with finish_reason=length
        assert content, f'empty content, finish_reason={choice.finish_reason}'
        return content

    def map(self, fn, items):
        '''
        Calls fn(self, item) for every item concurrently and yields (item, result) as they finish.
        Failed items are reported and skipped, so callers persist what they got and are rerun to
        retry the rest.
        '''
        n_failed = 0
        with ThreadPoolExecutor(max_workers = self.workers) as pool:
            futures = {pool.submit(fn, self, item): item for item in items}
            for future in tqdm(as_completed(futures), total = len(futures)):
                try:
                    yield futures[future], future.result()
                except Exception as e:
                    n_failed += 1
                    tqdm.write(f'[warn] {futures[future]!r:.80} failed: {type(e).__name__}: {e}')
        if n_failed:
            print(f'{n_failed}/{len(futures)} requests failed, rerun to retry them.')
