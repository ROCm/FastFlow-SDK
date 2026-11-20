# Run and serve models

FastFlowLM (`flm`) runs models on the NPU from the terminal or as a local
server with an OpenAI-compatible API, so existing OpenAI clients, LangChain,
and Open WebUI work unchanged. The model tags are on the [Models](reference/models.md)
page; `flm list` shows the ones in your installed release.

## Chat in the terminal

```bash
flm run qwen3:0.6b
```

The first run downloads the model from Hugging Face. Type `/bye` to exit.
`flm pull <tag>` downloads without starting a chat.

## Start the server

```bash
flm serve qwen3:0.6b
```

The server listens on `http://127.0.0.1:52625`, with the OpenAI API under
`/v1`. Change the port for one run with `--port <port>`, or for every run with
the `FLM_SERVE_PORT` environment variable. Other useful options:

| Option | Effect |
| --- | --- |
| `--ctx-len <tokens>` | Context length, rounded up to a power of two, at least 512 |
| `--pmode <mode>` | NPU power mode: `powersaver`, `balanced`, `performance` (default), or `turbo` |
| `--embed 1` | Also load the default embedding model, EmbeddingGemma; or pass an embedding model's tag |
| `--asr 1` | Also load the default speech-to-text model, Whisper; or pass a speech model's tag |
| `--cors 0` | Disable CORS, which is on by default; do this if the server is reachable beyond localhost |

One model of each kind (LLM, embedding, speech) can be loaded at a time, and
they can run together in one server.

## Endpoints

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | List the models |
| `POST /v1/chat/completions` | Chat, streaming, tool calls, and structured output |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/embeddings` | Embeddings; start the server with `--embed 1` |
| `POST /v1/audio/transcriptions` | Speech to text; start the server with `--asr 1` |

The server does not check API keys, but OpenAI clients require one, so pass
any string.

```bash
curl http://127.0.0.1:52625/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "Hello"}]}'
```

The Python examples below need `python -m pip install openai`. CI runs each one
except speech to text against `flm serve` on an NPU. Download them from this
site, for example
`curl -fsSLO https://rocm.github.io/FastFlow-SDK/examples/serve/chat.py`.

## Chat with streaming

```python
--8<-- "examples/serve/chat.py"
```

## Tool calls

The server parses each model's own tool-call markup into OpenAI `tool_calls`.
It supports `tool_choice` (`auto`, `required`, `none`, or one function) and
`parallel_tool_calls`. Use a model whose capabilities on
[Models](reference/models.md) include `tool-calling`, such as `gemma4-it:e2b`.

```python
--8<-- "examples/serve/tool_call.py"
```

## Thinking and structured output

Reasoning models put their thinking in `reasoning_content`, separate from
`content`. Turn thinking on or off per request with the FastFlowLM `think`
field, which the OpenAI SDK sends through `extra_body`:

```python
client.chat.completions.create(model="qwen3:0.6b", messages=messages, extra_body={"think": False})
```

The setting stays on the loaded model for later requests, so send it on every
request if you mix both. `response_format` accepts `json_object` and
`json_schema`, and constrains the output to the schema. FastFlowLM documents
both in [Reasoning, tool calls and streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/)
and [Structured output](https://fastflowlm.com/docs/instructions/server/structured_output/).

## Embeddings

```python
--8<-- "examples/serve/embeddings.py"
```

## Speech to text

```python
--8<-- "examples/serve/transcribe.py"
```

## Use your own NPU kernels in a model

FastFlowLM loads model extensions: engines built outside `flm` that replace
parts of a model with your own NPU programs.
[Replace Gemma 4 GEMM and dequant with IRON](examples/iron-gemma4-gemm-dequant.md)
walks through one.
