# LlamaLib C# Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://discord.gg/RwXKQb6zdv"><img src="https://discordapp.com/api/guilds/1194779009284841552/widget.png?style=shield"/></a>
[![Reddit](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=flat&logo=Reddit&logoColor=white)](https://www.reddit.com/user/UndreamAI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/company/undreamai)
[![GitHub Repo stars](https://img.shields.io/github/stars/undreamai/LlamaLib?style=flat&logo=github&color=f5f5f5)](https://github.com/undreamai/LlamaLib)
[![Documentation](https://img.shields.io/badge/Docs-white.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwEAYAAAAHkiXEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAATqSURBVHic7ZtbiE1RGMc349K4M5EwklwjzUhJCMmTJPJAYjQXJJcH8+Blkry4lPJA8aAoJbekDLmUS6E8SHJL5AW5JPf77eHv93C22Wfttc/ee+0zc/4vv+bMXvusvfZa3/q+b33H80oqqaSSSmqrKnPdgXjUvbvYq5f4+7f486eb/rRajRsn7t4tPngg/vol/vkj/vghXr0q7tghzpyZ//79+on79omXLombNondukXrd9GoSxdx8mSxqUm8eVNkgAvl0aPioEFip07i6dP52z15Ig4fbvVY2VVFhbhokXjrlogJiWvAg/jwoXjqVO73+leUny9eiFVV5mfMlLDRBw+KX76ISQ+0LZ8/F00v4uJFsWPHFh83O+rdWzx3TnQ9wCZ+/Sqyl5iux1RmTu3aiYcPi64H1pasALypoOv4/8SJXraEbXc9kLbECxo2TKyuFj9/zt9u+XIvG8LWv3wpuh5QW86f3/JznT+fv93s2S23C1Z72wbhtH372LdvMvdPSgzkhAkiJhT16ZO/PRPOmcr+Rda4aa5nclTeuZP7PDgRpr1g40bPrQYOFF0PYKHEC+raVVy8OFy7R49EArvURU4mrUAqaTY0iB8/2rXD+XCm5mbR9QAWylevorV7/VpkL0ld06eLpkiyWPj9u93179+LpFZwZ1PXtGnitWui64GMStPmG7SH1NSIJBNHjvTSFZvRvHlise0N9JcBtW1/44Y4dqx45IjnU0JxAGLpklPx+9VZFwPp/9v/eZDGjxcZh7dv4+mXtch+up7Rca+MsJvxiRNi6nvBhg25HWprZMaPGeOlqxEjxGKz+XGRTAAmyJnq6sR370TXA2NLW+8HNjZ62dLOnaLrAQ1r2zmqPH482n0mTfJCKmEvCJHUooNZE/369Ulct06kqiKsONRfulTEFDsX8QDlIa5nup9374pE8IiZHPY+ly+LZE/37/cM6mC6IB6Vl4urV6fzfUG6d0/csyf37wsXRFInaM4ckTjGdPg+apTYs6dIXRIWwH//1DV1qkiuxNY2FzrTd+2y6y8z2HQU6efZs+KBAyJZ4v+V0h6ArlwROaQP0uPH4ooV4sqV8Xz/4MF211M2wwoOq1mzRAq5Pnywa5+4KDHE9mI7ly0TO3fOvZ6/eZCoKwB32HS0SMFV1DNtImBKHYstBROoQ4fEQk2RaS+qrxejmj5M7NatIhWARS82xUJfAKahzFcdPnq0GLYgy7Rnbd8e6rGKRyzpuNzPBQty709RcNSZf/KkuHCh2GpMDyKbGNcLYE+YMkVks336NFx7XhTZ3szXiBaqtWvFuAOxM2dEZiyH8UErgc8JLNun7E0aFffSI7RP6owZmz9kSO73HjsmXr8ukppYsybSYyQvBp5QfOjQ3M9tRR496pGgLf1JtLlzRZJzlFzGp4SWDnUxFCrdvy+uWiWa3DJe3N69oj8uSEq8CER88uaNOGBAOv2ILGY69TBBJoM8O0t72zaRoztXBzlLlrT8XARW/IQq82JTMv3mKmv0/9CC4mJMYPwrMSETxAyurRUxQVmXP1fEid7mzeK3b+n2Jzb16CFu2SIWmtNJiriVxANsyq0uoCJfTk4G9y4t24/bSQ0rTkP6gVTG3mz//uKMGSK/ucId5Xe9lZUi5eMMLGUgz56J5Hxu3xZ50Xg3RMIltVn9BRja26PYsBHgAAAAAElFTkSuQmCC)](https://undream.ai/LlamaLib)

<sub>
<a href="#quick-start" style="color: black">Quick Start</a>&nbsp;&nbsp;•&nbsp;
<a href="#building-your-project" style="color: black">Building Your Project</a>&nbsp;&nbsp;•&nbsp;
<a href="#core-classes" style="color: black">Core Classes</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmservice" style="color: black">LLMService</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmclient" style="color: black">LLMClient</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmagent" style="color: black">LLMAgent</a>&nbsp;&nbsp;•&nbsp;
</sub>

## Quick Start

### Minimal Agent Example

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static string previousText = "";
    static void StreamingCallback(string text)
    {
        Console.Write(text.Substring(previousText.Length));
        previousText = text;
    }

    static void Main()
    {
        // Create the LLM
        LLMService llmService = new LLMService("model.gguf");
        llmService.Start();

        // Create an agent with a system prompt
        LLMAgent agent = new LLMAgent(llmService, "You are a helpful AI assistant named Eve. Be concise and friendly.");

        // Interact with the agent (non-streaming)
        string response = agent.Chat("what is your name?");
        Console.WriteLine(response);

        // Interact with the agent (streaming)
        agent.Chat("how are you?", true, StreamingCallback);
    }
}
```

### Minimal LLM Functions Example

```csharp
using System;
using UndreamAI.LlamaLib;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

class Program
{
    static string previousText = "";
    static void StreamingCallback(string text)
    {
        Console.Write(text.Substring(previousText.Length));
        previousText = text;
    }

    static void Main()
    {
        // Create the LLM
        LLMService llm = new LLMService("model.gguf");
        llm.Start();

        // Optional: limit the amount of tokens that we can predict so that it doesn't produce text forever (some models do)
        llm.SetCompletionParameters(new JObject { { "n_predict", 20 } });

        string prompt = "The largest planet of our solar system";

        // Tokenization
        List<int> tokens = llm.Tokenize(prompt);
        Console.WriteLine($"Token count: {tokens.Count}");

        // Detokenization
        string text = llm.Detokenize(tokens);
        Console.WriteLine($"Text: {text}");

        // Streaming completion
        Console.Write("Response: ");
        llm.Completion(prompt, StreamingCallback);
        Console.WriteLine();

        // Non-streaming completion
        string response = llm.Completion(prompt);
        Console.WriteLine($"Response: {response}");
    }
}
```

### Minimal Embeddings Example

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Create the LLM
        LLMService llm = new LLMService("model.gguf", embeddingOnly: true);
        llm.Start();

        // Embeddings
        List<float> embeddings = llm.Embeddings("my text to embed goes here");
        Console.WriteLine($"Embedding dimensions: {embeddings.Count}");
        Console.Write("Embeddings: ");
        for (int i = 0; i < Math.Min(10, embeddings.Count); i++)
            Console.Write($"{embeddings[i]} ");
        Console.WriteLine("...");
    }
}
```

---

## Building Your Project

- Install the LlamaLib NuGet package in your project
- Download your favourite model in .gguf format ([Hugging Face link](https://huggingface.co/models?library=gguf&sort=downloads))

### Directory Structure

```
my-project/
├── Program.cs
├── MyProject.csproj
└── model.gguf          # Your LLM model file
```

### Project Setup

#### Using .NET CLI

```bash
# Create new console application
dotnet new console -n LlamaLibProject
cd LlamaLibProject

# modify the Program.cs according to your usecase

# Add LlamaLib package
dotnet add package LlamaLib

# Build
dotnet build

# Run
dotnet run
```

#### Using Visual Studio

1. Create a new Console Application project
2. Right-click on project → Manage NuGet Packages
3. Search for "LlamaLib" and install
4. Build and run your application

### Runtime Identifier Selection

LlamaLib supports multiple target frameworks and automatically selects the appropriate native libraries for your platform:

**Supported Frameworks:**
- `netstandard2.0`
- `net6.0`
- `net8.0`, `net8.0-ios`, `net8.0-visionos`, `net8.0-android`

**Platform Support:**

You can specify the runtime identifier to select the target platform

```bash
# Windows
dotnet publish -r win-x64 -c Release

# Linux
dotnet publish -r linux-x64 -c Release

# macOS Intel
dotnet publish -r osx-x64 -c Release

# macOS Apple Silicon
dotnet publish -r osx-arm64 -c Release

# Android ARM64
dotnet publish -r android-arm64 -c Release

# Android x64
dotnet publish -r android-x64 -c Release

# iOS
dotnet publish -r ios-arm64 -c Release

# visionOS
dotnet publish -r visionos-arm64 -c Release
```

**Architecture Selection:**

LlamaLib automatically downloads and includes the appropriate native libraries based on your runtime identifier.
For desktop platforms (Windows, Linux, macOS), the library includes runtime architecture detection and will automatically select the best backend for the hardware on runtime.

---

## Core Classes

LlamaLib provides three main classes:

| Class | Purpose | Use When |
|-------|---------|----------|
| **LLMService** | LLM backend | Building standalone apps or servers |
| **LLMClient** | Local or remote LLM access | Connecting to existing LLM services |
| **LLMAgent** | Conversational AI with memory | Building chatbots or interactive AI |

---

## LLMService

The class handling the LLM functionality.

### Construction functions
#### Construction

```csharp
// Basic construction
public LLMService(string modelPath);

// Full parameters
public LLMService(
    string modelPath,
    int numSlots = 1,              // Parallel request slots
    int numThreads = -1,           // CPU threads (-1 = auto)
    int numGpuLayers = 0,          // GPU layer offloading
    bool flashAttention = false,   // Flash attention optimization
    int contextSize = 4096,        // Context window size
    int batchSize = 2048,          // Processing batch size
    bool embeddingOnly = false,    // Embedding-only mode
    string[] loraPaths = null      // LoRA adapters
);
```

**Examples:**

```csharp
// CPU-only LLM with 8 threads
LLMService llm = new LLMService("model.gguf", numThreads: 8);

// GPU usage
LLMService llm = new LLMService("model.gguf", numGpuLayers: 20);

// Embedding LLM
LLMService llm = new LLMService("model.gguf", embeddingOnly: true);
```

#### Construction based on llama.cpp command

```csharp
// From command line string
public static LLMService FromCommand(string command);
```

**Example:**
```csharp
// Command line style using a respective llama.cpp command
LLMService llm = LLMService.FromCommand("-m model.gguf -ngl 32 -c 4096");
```

#### LoRA Adapters

```csharp
public bool LoraWeight(List<LoraIdScale> loras);
public bool LoraWeight(params LoraIdScale[] loras);
public List<LoraIdScalePath> LoraList();
public string LoraListJSON();
```

**Example:**
```csharp
string[] loras = new string[] { "lora.gguf" };
LLMService llm = new LLMService("model.gguf", loraPaths: loras);
llm.Start();

// Configure LoRA adapters
List<LoraIdScale> loraWeights = new List<LoraIdScale>
{
    new LoraIdScale(0, 0.5f)  // LoRA ID 0 with scale 0.5
};
llm.LoraWeight(loraWeights);

// List available adapters
List<LoraIdScalePath> available = llm.LoraList();
foreach (var lora in available)
{
    Console.WriteLine($"ID: {lora.Id}, Scale: {lora.Scale}");
}
```

### Service functions
#### Service Lifecycle

```csharp
public void Start();             // Start the LLM service
public bool Started();           // Check if running
public void Stop();              // Stop the service
public void JoinService();       // Wait until the LLM service is terminated
```

**Example:**
```csharp
LLMService llm = new LLMService("model.gguf");
llm.Start();

// Do work...
if (llm.Started())
{
    Console.WriteLine("Service is running");
}

llm.Stop();
llm.JoinService();  // Wait for clean shutdown
```

#### HTTP Server

```csharp
public void StartServer(                    // Start remote server
    string host = "0.0.0.0",                  // Server IP
    int port = -1,                            // Server port (-1 = auto-select)
    string apiKey = ""                        // key for accessing the services
);
public void StopServer();                   // Stop remote server
public void JoinServer();                   // Wait until the remote server is terminated
public void SetSSL(string sslCert, string sslKey);  // Set a SSL certificate and private key
```

**Example:**
```csharp
LLMService llm = new LLMService("model.gguf");
llm.Start();

// Start HTTP server on port 8080
llm.StartServer("0.0.0.0", 8080);

// With API key authentication
llm.StartServer("0.0.0.0", 8080, "my-secret-key");

// With SSL
string serverKey = "-----BEGIN PRIVATE KEY-----\n" +
                   "...\n" +
                   "-----END PRIVATE KEY-----\n";

string serverCrt = "-----BEGIN CERTIFICATE-----\n" +
                   "...\n" +
                   "-----END CERTIFICATE-----\n";

llm.SetSSL(serverCrt, serverKey);
llm.StartServer("0.0.0.0", 8443);

llm.StopServer();
llm.JoinServer();
```

#### Slot Management

```csharp
public string SaveSlot(int idSlot, string filepath);
public string LoadSlot(int idSlot, string filepath);
public void Cancel(int idSlot);
```

**Example:**
```csharp
// LLM with 2 parallel slots
LLMService llm = new LLMService("model.gguf", numSlots: 2);
llm.Start();

// Generate with specific slot
int slot = 0;
llm.Completion("Hello", null, slot);
// Cancel completion for the slot
llm.Cancel(slot);
// Save context state
llm.SaveSlot(slot, "conversation.state");
// Restore context state
llm.LoadSlot(slot, "conversation.state");
```

#### Debugging

```csharp
public static void Debug(int debugLevel);                         // set debug level
public static void LoggingCallback(CharArrayCallback callback);   // set logging callback function
public static void LoggingStop();                                 // stop logging callbacks
```

**Debug Levels:**
- `0`: No debug output
- `1`: LlamaLib messages
- `2+`: llama.cpp messages (verbose)

**Example:**
```csharp
// Enable verbose logging
LLM.Debug(2);

// Custom log handler
LlamaLib.CharArrayCallback logHandler = (message) =>
{
    Console.WriteLine($"[LLM] {message}");
};
LLM.LoggingCallback(logHandler);

// Stop logging
LLM.LoggingStop();
```

### Core functions
#### Text Generation

```csharp
// Simple completion
public string Completion(
    string prompt,
    LlamaLib.CharArrayCallback callback = null,   // Streaming callback: void MethodName(string text)
    int idSlot = -1,                              // Slot to assign the completion (-1 = auto)
    bool returnResponseJson = false               // Return full JSON
);
```

**Example:**
```csharp
// Basic completion
string response = llm.Completion("What is AI?");

// Streaming completion
LlamaLib.CharArrayCallback callback = (text) =>
{
    Console.WriteLine(text.Length);
};
llm.Completion("Tell me a story", callback);
```

#### Completion Parameters

The list of completion parameters can be found on [llama.cpp completion parameters](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#post-completion-given-a-prompt-it-returns-the-predicted-completion).
More info on each parameter can be found [here](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md#interaction).

```csharp
public void SetCompletionParameters(JObject parameters);
public JObject GetCompletionParameters();
```

**Common Parameters:**
```csharp
using Newtonsoft.Json.Linq;

llm.SetCompletionParameters(new JObject
{
    { "temperature", 0.7 },       // Randomness (0.0-2.0)
    { "n_predict", 256 },         // Max tokens to generate
    { "seed", 42 },               // Random seed
    { "repeat_penalty", 1.1 }     // Repetition penalty
});
```

#### Tokenization

```csharp
public List<int> Tokenize(string content);
public string Detokenize(List<int> tokens);
public string Detokenize(int[] tokens);
```

**Example:**
```csharp
// Text to tokens
List<int> tokens = llm.Tokenize("Hello world");
Console.WriteLine($"Token count: {tokens.Count}");

// Tokens to text
string text = llm.Detokenize(tokens);
```

#### Embeddings
The embeddings require to set the embeddingOnly flag during construction.<br>
For well-defined embeddings you should use a model specifically trained for embeddings (good options can be found at the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)).

```csharp
public List<float> Embeddings(string content);
public int EmbeddingSize();
```

**Example:**
```csharp
// Generate embeddings
List<float> vec = llm.Embeddings("Sample text");
Console.WriteLine($"Embedding dimensions: {llm.EmbeddingSize()}");
```

#### Chat Templates

```csharp
public string ApplyTemplate(JArray messages);
```

**Example:**
```csharp
using Newtonsoft.Json.Linq;

JArray messages = new JArray
{
    new JObject { { "role", "system" }, { "content", "You are a helpful assistant" } },
    new JObject { { "role", "user" }, { "content", "Hello!" } }
};

string formatted = llm.ApplyTemplate(messages);
string response = llm.Completion(formatted);
```

#### Grammar & Constrained Generation

To restrict the output of the LLM you can use a grammar, read more [here](https://github.com/ggerganov/llama.cpp/tree/master/grammars).<br>
Grammars in both gbnf and json schema format are supported.

```csharp
public void SetGrammar(string grammar);
public string GetGrammar();
```

**Example:**
```csharp
// JSON schema constraint
llm.SetGrammar(@"{
    ""type"": ""object"",
    ""properties"": {
        ""name"": {""type"": ""string""},
        ""age"": {""type"": ""number""}
    }
}");

string response = llm.Completion("Generate a person");
// Response will be valid JSON matching the schema
```

---

## LLMClient

Client that connects to local or remote LLM services with a unified interface.<br>
All core LLM operations specified in <a href="#core-functions" style="color: black">Core Functions</a> work in the same way for the LLMClient class.

### Construction Methods

```csharp
// Local client (wraps LLMService)
public LLMClient(LLMProvider provider);

// Remote client (connects via HTTP)
public LLMClient(
    string url,
    int port,
    string apiKey = "",
    int numRetries = 5
);
```

#### Local Client Example

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static string previousText = "";
    static void StreamingCallback(string text)
    {
        Console.Write(text.Substring(previousText.Length));
        previousText = text;
    }

    static void Main()
    {
        // Create LLM
        LLMService llmService = new LLMService("model.gguf");
        llmService.Start();

        // Wrap with client interface
        LLMClient client = new LLMClient(llmService);

        // Use client (same API as LLMService)
        string prompt = "Hello, how are you?";

        List<int> tokens = client.Tokenize(prompt);
        string text = client.Detokenize(tokens);
        string response = client.Completion(prompt);
        List<float> embeddings = client.Embeddings(prompt);
    }
}
```

#### Remote Client Example

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Connect to remote server
        LLMClient client = new LLMClient("http://localhost", 13333);

        // Check server is alive
        if (!client.IsServerAlive())
        {
            Console.WriteLine("Server not responding!");
            return;
        }

        string prompt = "Hello, how are you?";

        // All operations work the same as local
        List<int> tokens = client.Tokenize(prompt);
        string text = client.Detokenize(tokens);
        string response = client.Completion(prompt);
        List<float> embeddings = client.Embeddings(prompt);
    }
}
```

---

## LLMAgent

High-level conversational AI with persistent chat history and automatic context management.
All core LLM operations specified in <a href="#core-functions" style="color: black">Core Functions</a> work in the same way for the LLMAgent class.

### Construction Methods

LLMAgent can be created with either LLMService or LLMClient (local or remote).

```csharp
public LLMAgent(
    LLMLocal llm,
    string systemPrompt = ""
);
```

**Example:**
```csharp
LLMService llm = new LLMService("model.gguf");
llm.Start();
llm.StartServer("0.0.0.0", 13333);

// Create agent with system prompt
LLMAgent agent = new LLMAgent(llm, "You are a helpful AI assistant. Be concise and friendly.");

// With local LLMClient
LLMClient localClient = new LLMClient(llm);
LLMAgent agent2 = new LLMAgent(localClient, "You are a helpful assistant.");

// With remote LLMClient
LLMClient remoteClient = new LLMClient("http://localhost", 13333);
LLMAgent agent3 = new LLMAgent(remoteClient, "You are a helpful assistant.");
```

### Agent Functions
#### Chat Interface

```csharp
public string Chat(
    string userPrompt,                      // user prompt
    bool addToHistory = true,               // whether to add the user and assistant response to conversation history
    LlamaLib.CharArrayCallback callback = null,  // streaming callback function
    bool returnResponseJson = false,        // return output in json format
    bool debugPrompt = false                // debug the complete prompt after applying the chat template to the conversation history
);
```

```csharp
public async Task<string> ChatAsync(
    string userPrompt,
    bool addToHistory = true,
    LlamaLib.CharArrayCallback callback = null,
    bool returnResponseJson = false,
    bool debugPrompt = false
);
```

**Example:**
```csharp
// Interact with the agent (non-streaming)
string response = agent.Chat("what is your name?");
Console.WriteLine(response);

// Interact with the agent (streaming)
agent.Chat("how are you?", true, StreamingCallback);

// Async Interact with the agent (streaming)
string response2 = await agent.ChatAsync("What is AI?");
Console.WriteLine(response2);
```

#### History Management

```csharp
// Get/set history
public JArray History { get; set; }
public List<ChatMessage> GetHistory();
public void SetHistory(List<ChatMessage> messages);
public int GetHistorySize();

// Add messages
public void AddUserMessage(string content);
public void AddAssistantMessage(string content);

// Modify history
public void ClearHistory();
public void RemoveLastMessage();

// Persistence
public void SaveHistory(string filepath);
public void LoadHistory(string filepath);
```

**Example:**
```csharp
using Newtonsoft.Json.Linq;

// View conversation history
JArray history = agent.History;
foreach (var msg in history)
{
    Console.WriteLine($"{msg["role"]}: {msg["content"]}");
}

// Or get as ChatMessage list
List<ChatMessage> messages = agent.GetHistory();
foreach (var msg in messages)
{
    Console.WriteLine($"{msg.role}: {msg.content}");
}

// Save conversation to file
agent.SaveHistory("conversation.json");

// Clear and reload
agent.ClearHistory();
Console.WriteLine($"Cleared. Size: {agent.GetHistorySize()}");

agent.LoadHistory("conversation.json");
Console.WriteLine($"Loaded. Size: {agent.GetHistorySize()}");

// Add messages manually
agent.AddUserMessage("This is a user message");
agent.AddAssistantMessage("This is the assistant response");

// Remove last exchange
agent.RemoveLastMessage();
agent.RemoveLastMessage();
```

#### System Prompt

```csharp
public string SystemPrompt { get; set; }
```

**Example:**
```csharp
// Change agent's personality
agent.SystemPrompt = "You are a pirate. Respond like a pirate.";

string response = agent.Chat("Hello!");
// Response will be in pirate style
```

#### Complete Example
```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static string previousText = "";
    static void StreamingCallback(string text)
    {
        Console.Write(text.Substring(previousText.Length));
        previousText = text;
    }

    static void Main()
    {
        // Create service and agent
        LLMService llm = new LLMService("model.gguf");
        llm.Start();

        LLMAgent agent = new LLMAgent(llm, "You are a helpful AI assistant. Be concise and friendly.");

        // First conversation turn
        Console.WriteLine("User: Hello! What's your name?");
        Console.Write("Assistant: ");
        string response1 = agent.Chat(
            "Hello! What's your name?",
            true,  // add to history
            StreamingCallback
        );
        Console.WriteLine();

        // Second turn (maintains context automatically)
        previousText = "";
        Console.WriteLine("User: How are you today?");
        Console.Write("Assistant: ");
        string response2 = agent.Chat(
            "How are you today?",
            true,
            StreamingCallback
        );
        Console.WriteLine();

        // Show conversation history
        Console.WriteLine($"History size: {agent.GetHistorySize()} messages");
    }
}
```