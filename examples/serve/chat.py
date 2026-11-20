#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Chat with a model that FastFlowLM serves on the NPU, streaming the reply.

Start the server in another terminal first:

    flm serve qwen3:0.6b
    python -m pip install openai
    python examples/serve/chat.py
"""

import os

from openai import OpenAI

# flm serve listens on 52625 unless FLM_SERVE_PORT or --port changes it.
BASE_URL = os.environ.get("FLM_BASE_URL", "http://127.0.0.1:52625/v1")
MODEL = os.environ.get("FLM_MODEL", "qwen3:0.6b")

# FastFlowLM does not check the key, but the client requires one.
client = OpenAI(base_url=BASE_URL, api_key="flm")

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "In one sentence, what is an NPU?"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
