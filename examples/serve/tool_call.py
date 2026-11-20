#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Let a model that FastFlowLM serves call a Python function.

Use a model whose Capabilities on the Models page include tool calling:

    flm serve gemma4-it:e2b
    python examples/serve/tool_call.py
"""

import json
import os

from openai import OpenAI

BASE_URL = os.environ.get("FLM_BASE_URL", "http://127.0.0.1:52625/v1")
MODEL = os.environ.get("FLM_MODEL", "gemma4-it:e2b")

client = OpenAI(base_url=BASE_URL, api_key="flm")


def get_weather(city: str) -> dict:
    return {"city": city, "forecast": "sunny", "celsius": 21}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get today's weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What is the weather in Austin today?"}]
reply = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
message = reply.choices[0].message
messages.append(message.model_dump(exclude_none=True))
for call in message.tool_calls or []:
    print(f"tool call: {call.function.name}({call.function.arguments})")
    result = get_weather(**json.loads(call.function.arguments))
    messages.append(
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
    )

final = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
print(final.choices[0].message.content)
