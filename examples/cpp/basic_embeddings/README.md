# Basic Embeddings Example

## What it does

The basic_embeddings example demonstrates how to generate vector embeddings for text.

## How to Build
1. Download and extract the LlamaLib release LlamaLib-vX.X.X.zip from the [latest release](https://github.com/undreamai/LlamaLib/releases/latest)
2. Build with CMake:

```bash
cmake -B build -DLlamaLib_DIR=<path_to_extracted_LlamaLib_release>
cmake --build build
```

## How to Run
1. Copy an embedding model .gguf file as `model.gguf` inside the directory

2. Run
```bash
./build/main
```
