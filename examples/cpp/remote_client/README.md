# Remote Client Example

## What it does

The remote client example demonstrates the remote server-client functionality of the `LLMClient` class.  
The `LLMClient` class in remote mode connects to an LLM server running on a different process or machine via HTTP. This example demonstrates:

- **Client-Server Architecture**: Separating LLM processing from client application
- **Server Management**: Starting/stopping a server for demonstration (server.cpp)
- **Remote Client Access**: Connecting to an LLM server via HTTP (main.cpp)
- **Network Communication**: All operations performed through HTTP requests (main.cpp)

This is useful for distributed applications, web services, or when you want to share LLM resources across multiple client applications.

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
1. Copy a model .gguf file as `model.gguf` inside the directory (or change the `model = ...` line in server.cpp)

2. Run the server
```bash
./server
```

3. Run the client in a second terminal
```bash
./main
```
