# Remote Client Example

## What it does

The remote client example demonstrates the remote server-client functionality:
- **Client-Server Architecture**: Separating LLM processing from client application
- **Server Management**: Starting a separater server (server.cpp)
- **Remote Client**: Connecting a client to an LLM server (main.cpp)
- **Remote Agent**: Creating an agent based on the client (main.cpp)

## How to Build
1. Download and extract the LlamaLib release LlamaLib-vX.X.X.zip from the [latest release](https://github.com/undreamai/LlamaLib/releases/latest)
2. Build with CMake:

```bash
cmake -B build -DLlamaLib_DIR=<path_to_extracted_LlamaLib_release>
cmake --build build
```

## How to Run
1. Copy a model .gguf file as `model.gguf` inside the directory

2. Run the server
```bash
./build/server
```

3. Run the client in a second terminal
```bash
./build/main
```
