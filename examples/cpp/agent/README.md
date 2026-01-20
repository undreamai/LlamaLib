# Agent Example

## What it does

The agent example demonstrates how to setup and interact with an agents as well as access the agent history.

## How to Build
1. Download and extract the LlamaLib release LlamaLib-vX.X.X.zip from the [latest release](https://github.com/undreamai/LlamaLib/releases/latest)
2. Build with CMake:

```bash
cmake -B build -DLLAMALIB_DIR=<path_to_extracted_LlamaLib_release>
cmake --build build
```

## How to Run
1. Copy a model .gguf file as `model.gguf` inside the directory

2. Run
```bash
./build/main
```
