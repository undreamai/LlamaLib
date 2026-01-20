# Remote Client Example

## What it does

The remote client example demonstrates the remote server-client functionality:
- **Client-Server Architecture**: Separating LLM processing from client application
- **Server Management**: Starting a separater server (server.cpp)
- **Remote Client**: Connecting a client to an LLM server (main.cpp)
- **Remote Agent**: Creating an agent based on the client (main.cpp)

## How to Build
```bash
dotnet build -c Server
dotnet build -c Client
```

## How to Run
1. Copy a model .gguf file as `model.gguf` inside the directory

2. Run the server
```bash
dotnet run -c Server
```

3. Run the client in a second terminal
```bash
dotnet run -c Client
```