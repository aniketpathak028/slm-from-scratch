# tinystories small language model

A small language model (SLM) with ~50M parameters trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and served entirely in the browser using WebAssembly.

---

## Features

- **In-Browser Inference:** Runs locally on the user's CPU via ONNX Runtime WebAssembly (zero server cost).
- **Quantized Weights:** Model weights quantized to `int8` (~47 MB) for fast web loading.
- **ChatGPT-Style UI:** Clean, responsive dark mode interface with real-time word streaming.
- **Generation Controls:** Includes auto-resizing input, stop generation mid-stream, and prompt resetting.

---

## Model Details

- **Parameters:** ~50–60 Million
- **Architecture:** GPT-style Decoder-only Transformer (6 Layers, 6 Heads, 384 Embedding Dim)
- **Context Window:** 128 tokens
- **Tokenizer:** GPT-2 BPE (`tiktoken`)
- **Dataset:** TinyStories
