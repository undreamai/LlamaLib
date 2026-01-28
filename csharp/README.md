# LlamaLib

Cross-platform C# library for running Large Language Models (LLMs) locally in your applications.

## Features

- **High-Level API** - Intuitive object-oriented C# design
- **Self-Contained and Embedded** - No external dependencies, servers, or open ports required
- **Runs Anywhere** - Windows, macOS, Linux, Android, iOS, Meta Quest, Apple Vision
- **Architecture Detection at runtime** - Automatically selects optimal CPU/GPU backend
- **Small footprint** - Around 100 MB for CPU, GPU support adds 70MB-1.3GB depending on backend
- **Production ready** - Easy integration, supports both local and client-server deployment.

## Installation

```bash
dotnet add package LlamaLib
```

Or via NuGet Package Manager:
```
Install-Package LlamaLib
```

## Documentation

- [API Guide](https://github.com/undreamai/LlamaLib/tree/main/csharp_api.md)
- [Examples](https://github.com/undreamai/LlamaLib/tree/main/examples/csharp)
- [GitHub Repository](https://github.com/undreamai/LlamaLib)

### Core classes

LlamaLib provides three main classes for different use cases:

| Class | Purpose | Best For |
|-------|---------|----------|
| **LLMService** | LLM backend engine | Building standalone apps or servers |
| **LLMClient** | Local or remote LLM access | Connecting to existing LLM services |
| **LLMAgent** | Conversational AI with memory | Building chatbots or interactive AI |


## Example

```csharp
using LlamaLib;

class Program {
    static void Main() {
        // Same API, different language
        LLMService llm = new LLMService("path/to/model.gguf");
        /* Optional parameters:
           threads=-1,     // CPU threads (-1 = auto)
           gpu_layers=0,   // GPU layers (0 = CPU only)
           num_slots=1     // parallel slots/clients
        */
        
        llm.Start();
        
        string response = llm.Completion("Hello, how are you?");
        Console.WriteLine(response);
        
        // Supports streaming operation to your function:
        // llm.Completion(prompt, streamingCallback);
    }
}
```

## Support

- Join our [Discord](https://discord.gg/RwXKQb6zdv) community
- Report issues on [GitHub](https://github.com/undreamai/LlamaLib/issues)
- [Sponsor](https://github.com/sponsors/amakropoulos) development or support with a [Ko-fi](https://ko-fi.com/amakropoulos)

## License

Apache 2.0