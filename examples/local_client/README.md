# Local Client Example

## What it does

The local client example demonstrates the local functionality of the `LLMClient` class.  
The `LLMClient` class creates a client that accesses a `LLMService` instance providing the LLM functionality.  
In local mode, it takes as input a `LLMService` instance to provide the same operations through a consistent client interface. This example demonstrates:

- **Local Client Access**: Using LLMClient to access a local LLMService
- **Unified Interface**: Same API whether accessing local or remote LLMs
- **All LLM Operations**: Tokenization, completion, embeddings through client interface

The local client is useful when you want to create multiple clients to access a single LLM service.

## How to Build
1. Download and extract the LlamaLib release LlamaLib-vX.X.X.zip from the [latest release](https://github.com/undreamai/LlamaLib/releases/latest)
2. Build with CMake:

```bash
mkdir build
cd build
cmake .. -DLlamaLib_DIR=<path_to_extracted_LlamaLib_release>
make
```

## How to Run
1. Copy a model .gguf file as `model.gguf` inside the directory (or change the `model = ...` line in main.cpp)

2. Run
```bash
./main
```
