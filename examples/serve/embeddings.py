#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Embed text with EmbeddingGemma, served by FastFlowLM beside an LLM.

flm serve qwen3:0.6b --embed 1
python examples/serve/embeddings.py
"""

import math
import os

from openai import OpenAI

BASE_URL = os.environ.get("FLM_BASE_URL", "http://127.0.0.1:52625/v1")

client = OpenAI(base_url=BASE_URL, api_key="flm")


def embed(text: str) -> list[float]:
    return client.embeddings.create(model="embed-gemma", input=text).data[0].embedding


a = embed("The model runs on the NPU.")
b = embed("The NPU executes the model.")
cosine = sum(x * y for x, y in zip(a, b, strict=True)) / (
    math.hypot(*a) * math.hypot(*b)
)
print(f"{len(a)} dimensions, cosine similarity {cosine:.3f}")
