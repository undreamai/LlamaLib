# Agent Example

## What it does

The agent example demonstrates the agent functionality of the `LLMAgent` class.  
The `LLMAgent` class provides a high-level conversational interface that manages chat history and applies chat templates automatically. This example demonstrates:

- **Conversation Management**: Maintaining chat history across multiple turns
- **System Prompts**: Setting initial context for the conversation
- **Role Management**: Managing user, assistant, and system roles
- **History Persistence**: Saving/loading conversation history to/from files
- **Slot Operations**: Saving/restoring the agent's processing state
- **Template Application**: Automatic application of chat templates

LLMAgent is ideal for building chatbots, conversational AI applications, or any use case where you need to maintain conversation context.

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
