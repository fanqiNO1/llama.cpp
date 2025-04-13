## Kimi-VL

### Prepare models and code

Download [Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) or [Kimi-VL-A3B-Thinking](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) PyTorch model from huggingface to "Kimi-VL-A3B-Instruct" or "Kimi-VL-A3B-Thinking" folder.

### Build llama.cpp

Readme modification time: 20250411

If there are differences in usage, please refer to the official build [documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

Clone llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

### Usage of Kimi-VL-A3B-Instruct

The usage of Kimi-VL-A3B-Thinking is similar.

Convert PyTorch model to gguf files

```bash
TODO
```

Inference on Linux or Mac

```bash
TODO
```