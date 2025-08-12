# Basic Example

## What it does

The basic example demonstrates the `LLMService` class.  
The `LLMService` class is the core class that provides the actual LLM processing capabilities.  
The `LLMService` loads and serves the LLM model.  
This example demonstrates:

- **Model Loading**: Loading a GGUF model file directly
- **Tokenization**: Converting text to token IDs and back
- **Text Completion**: Generating text responses (both streaming and non-streaming)
- **Embeddings**: Generating vector embeddings for text

## How to Build
```bash
dotnet build
```

## How to Run
1. Copy a model .gguf file as `model.gguf` inside the directory (or change the `model = ...` line in Program.cs)

2. Run
```bash
dotnet run
```
