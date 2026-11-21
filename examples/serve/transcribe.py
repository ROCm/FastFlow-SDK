#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Transcribe an audio file with Whisper, served by FastFlowLM.

flm serve --asr 1
python examples/serve/transcribe.py recording.mp3
"""

import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("FLM_BASE_URL", "http://127.0.0.1:52625/v1")

client = OpenAI(base_url=BASE_URL, api_key="flm")

with open(sys.argv[1], "rb") as audio:
    print(client.audio.transcriptions.create(model="whisper-v3", file=audio).text)
