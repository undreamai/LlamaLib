<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset=".github/logo.png">
  <img src=".github/logo.png" height="150"/>
</picture>
</p>

<h3 align="center">Cross-Platform High-Level LLM Library</h3>

[![License: Apache](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
<a href="https://discord.gg/RwXKQb6zdv"><img src="https://discordapp.com/api/guilds/1194779009284841552/widget.png?style=shield"/></a>
[![Reddit](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=flat&logo=Reddit&logoColor=white)](https://www.reddit.com/user/UndreamAI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/company/undreamai)
[![GitHub Repo stars](https://img.shields.io/github/stars/undreamai/LlamaLib?style=flat&logo=github&color=f5f5f5)](https://github.com/undreamai/LlamaLib)
[![Documentation](https://img.shields.io/badge/Docs-white.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwEAYAAAAHkiXEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAATqSURBVHic7ZtbiE1RGMc349K4M5EwklwjzUhJCMmTJPJAYjQXJJcH8+Blkry4lPJA8aAoJbekDLmUS6E8SHJL5AW5JPf77eHv93C22Wfttc/ee+0zc/4vv+bMXvusvfZa3/q+b33H80oqqaSSSmqrKnPdgXjUvbvYq5f4+7f486eb/rRajRsn7t4tPngg/vol/vkj/vghXr0q7tghzpyZ//79+on79omXLombNondukXrd9GoSxdx8mSxqUm8eVNkgAvl0aPioEFip07i6dP52z15Ig4fbvVY2VVFhbhokXjrlogJiWvAg/jwoXjqVO73+leUny9eiFVV5mfMlLDRBw+KX76ISQ+0LZ8/F00v4uJFsWPHFh83O+rdWzx3TnQ9wCZ+/Sqyl5iux1RmTu3aiYcPi64H1pasALypoOv4/8SJXraEbXc9kLbECxo2TKyuFj9/zt9u+XIvG8LWv3wpuh5QW86f3/JznT+fv93s2S23C1Z72wbhtH692LdvMvdPSgzkhAkiJhT16ZO/PRPOmcr+Rda4aa5nclTeuZP7PDgRpr1g40bPrQYOFF0PYKHEC+raVVy8OFy7R49EArvURU4mrUAqaTY0iB8/2rXD+XCm5mbR9QAWylevorV7/VpkL0ld06eLpkiyWPj9u93179+LpFZwZ1PXtGnitWui64GMStPmG7SH1NSIJBNHjvTSFZvRvHlise0N9JcBtW1/44Y4dqx45IjnU0JxAGLpklPx+9VZFwPp/9v/eZDGjxcZh7dv4+mXtch+up7Rca+MsJvxiRNi6nvBhg25HWprZMaPGeOlqxEjxGKz+XGRTAAmyJnq6sR370TXA2NLW+8HNjZ62dLOnaLrAQ1r2zmqPH482n0mTfJCKmEvCJHUooNZE/369Elct06kqiKsONRfulTEFDsX8QDlIa5nup9374pE8IiZHPY+ly+LZE/37/cM6mC6IB6Vl4urV6fzfUG6d0/csyf37wsXRFInaM4ckTjGdPg+apTYs6dI3RIWwH//1DV1qkiuxNY2FzrTd+2y6y8z2HQU6efZs+KBAyJZ4v+V0h6ArlwROaQP0uPH4ooV4sqV8Xz/4MF211M2wwoOq1mzRAq5Pnywa5+4KDHE9mI7ly0TO3fOvZ6/eZCoKwB32HS0SMFV1DNtImBKHYstBROoQ4fEQk2RaS+qrxejmj5M7NatIhWARS82xUJfAKahzFcdPnq0GLYgy7Rnbd8e6rGKRyzpuNzPBQty709RcNSZf/KkuHCh2GpMDyKbGNcLYE+YMkVks336NFx7XhTZ3szXiBaqtWvFuAOxM2dEZiyH8UErgc8JLNun7E0aFffSI7RP6owZmz9kSO73HjsmXr8ukppYsybSYyQvBp5QfOjQ3M9tRR496pGgLf1JtLlzRZJzlFzGp4SWDnUxFCrdvy+uWiWa3DJe3N69oj8uSEq8CER88uaNOGBAOv2ILGY69TBBJoM8O0t72zaRoztXBzlLlrT8XARW/IQq82JTMv3mKmv0/9CC4mJMYPwrMSETxAyurRUxQVmXP1fEid7mzeK3b+n2Jzb16CFu2SIWmtNJiriVxANsyq0uoCJfTk4G9y4t24/bSQ0rTkP6gVTG3mz//uKMGSK/ucId5Xe9lZUi5eMMLGUgz56J5Hxu3xZ50Xg3RMIltVn9BRja26PYsBHgAAAAAElFTkSuQmCC)](https://undream.ai/LlamaLib)

LlamaLib is a **high-level C++ and C#** library for running Large Language Models (LLMs) **anywhere** - from PCs to mobile devices and VR headsets.<br>
It is built on top of the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

---

## At a glance

- ✅ **High-Level API**  
  C++ and C# implementations with intuitive object-oriented design.

- 📦 **Self-Contained and Embedded**  
  Runs embedded within your application.  
  No need for a separate server, open ports or external processes.  
  Zero external dependencies.

- 🌍 **Runs Anywhere**  
  Cross-platform and cross-device.  
  Works on all major platforms:
    - Desktop: `Windows`, `macOS`, `Linux`
    - Mobile: `Android`, `iOS`
    - VR/AR: `Meta Quest`, `Apple Vision`, `Magic Leap`

  and hardware architectures:
    - CPU: Intel, AMD, Apple Silicon
    - GPU: NVIDIA, AMD, Metal

- 🔍 **Architecture Detection at runtime**  
  Automatically selects the optimal backend at runtime supporting all major GPU and CPU architectures.

- 💾 **Small footprint**  
  Integration requires around 100 MB for CPU architectures and offers GPU support with 70MB (Vulkan) / 370 MB (tinyBLAS) / 1.3 GB (cuBLAS).

- 🛠️ **Production ready**  
  Designed for easy integration into C++ and C# applications.  
  Supports both local and client-server deployment.

---

## Why LlamaLib?

### Developer API
- **Direct implementation** of LLM operations (completion, tokenization, embeddings)
- **Clean architecture** for services, clients, and agents
- **Simple server-client** setup with built-in SSL and authentication support

### Universal Deployment
- **The only library** that lets you build for any hardware with runtime detection unlike alternatives limited to specific GPU vendors or CPU-only execution
- **GPU backend auto-selection:** Automatically chooses NVIDIA, AMD, Metal or switch to CPU
- **CPU optimization:** Identifies and uses optimal CPU instruction sets

### Production Ready
- **Embedded deployment:** No need for open ports or external processes
- **Small footprint:** Compact builds ideal for PC or mobile deployment
- **Battle-tested:** Powers [LLM for Unity](https://github.com/undreamai/LLMUnity), the most widely used LLM integration for games

---

## How to help
- ⭐ [Star the repo](https://github.com/undreamai/LlamaLib) and spread the word!
- ❤️ [Sponsor](https://github.com/sponsors/amakropoulos) development or support with a [![Ko-fi](https://img.shields.io/badge/Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/amakropoulos)
- 💬 Join our [Discord](https://discord.gg/RwXKQb6zdv) community
- 🐛 [Contribute](CONTRIBUTING.md) with feature requests, bug reports, or pull requests

---

## Projects using LlamaLib
- [LLM for Unity](https://github.com/undreamai/LLMUnity): The most widely used solution to integrate LLMs in games

---

## Quick Start

### Documentation

**Language Guides:**
- **C++**: [API guide](cpp_api.md) • [Examples](examples/cpp)
- **C#**: [API guide](csharp_api.md) • [Examples](examples/csharp)

### Core classes

LlamaLib provides three main classes for different use cases:

| Class | Purpose | Best For |
|-------|---------|----------|
| **LLMService** | LLM backend engine | Building standalone apps or servers |
| **LLMClient** | Local or remote LLM access | Connecting to existing LLM services |
| **LLMAgent** | Conversational AI with memory | Building chatbots or interactive AI |

### C++ Example

```cpp
#include "LlamaLib.h"

int main() {
    // LlamaLib automatically detects your hardware and selects optimal backend
    LLMService llm("path/to/model.gguf");
    /* Optional parameters:
       threads=-1,     // CPU threads (-1 = auto)
       gpu_layers=0,   // GPU layers (0 = CPU only)
       num_slots=1     // parallel slots/clients
    */
    
    // Start service
    llm.start();
    
    // Generate completion
    std::string response = llm.completion("Hello, how are you?");
    std::cout << response << std::endl;
    
    // Supports streaming operation to your function:
    // llm.completion(prompt, streaming_callback);
    
    return 0;
}
```

**📖 See the [C++ guide](cpp_api.md) for installation, building, and complete API reference.**

### C# Example

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

**📖 See the [C# guide](csharp_api.md) for installation, NuGet setup, and complete API reference.**

---

## License

LlamaLib is licensed under the [Apache 2.0](LICENSE.md).