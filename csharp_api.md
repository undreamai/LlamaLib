# LlamaLib C# API Guide

Complete reference for using LlamaLib in your C# and .NET applications.

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
- [LLMService](#llmservice)
- [LLMClient](#llmclient)
- [LLMAgent](#llmagent)
- [Common Patterns](#common-patterns)
- [Advanced Features](#advanced-features)

---

## Getting Started

### Project Setup

LlamaLib for C# requires .NET and the Newtonsoft.Json package:

**Using .NET CLI:**
```bash
# Create a new console application
dotnet new console -n MyLLMApp
cd MyLLMApp

# Add required package
dotnet add package Newtonsoft.Json

# Add LlamaLib reference
# Copy LlamaLib.dll and llamalib.dll to your project directory
```

**Project Structure:**
```
MyLLMApp/
├── MyLLMApp.csproj
├── Program.cs
├── LlamaLib.dll          # C# wrapper
├── llamalib.dll          # Native library
└── model.gguf            # Your LLM model file
```

**MyLLMApp.csproj:**
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>

  <ItemGroup>
    <Reference Include="LlamaLib">
      <HintPath>LlamaLib.dll</HintPath>
    </Reference>
  </ItemGroup>

  <ItemGroup>
    <None Update="llamalib.dll">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="model.gguf">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
  </ItemGroup>
</Project>
```

### Required Namespaces

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;
```

---

## Quick Start

### Minimal Example

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Create and start LLM service
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        // Generate text
        string response = llm.Completion("Hello, how are you?");
        Console.WriteLine(response);
    }
}
```

### Complete Basic Example

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static void StreamingCallback(string text)
    {
        Console.Write(text);
    }

    static void Main()
    {
        string model = "model.gguf";
        using var llm = new LLMService(model);
        llm.Start();
        
        string prompt = "Hello, how are you?";
        
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
        
        // Embeddings
        List<float> embeddings = llm.Embeddings(prompt);
        Console.WriteLine($"Embedding dimensions: {embeddings.Count}");
    }
}
```

---

## Core Classes

LlamaLib provides three main classes:

| Class | Purpose | Use When |
|-------|---------|----------|
| **LLMService** | Complete LLM backend with server | Building standalone apps or servers |
| **LLMClient** | Local or remote LLM access | Connecting to existing LLM services |
| **LLMAgent** | Conversational AI with memory | Building chatbots or interactive AI |

**Class Hierarchy:**
```
LLM (abstract base)
├── LLMLocal (abstract)
│   ├── LLMProvider (abstract)
│   │   └── LLMService
│   ├── LLMClient
│   └── LLMAgent
```

---

## LLMService

The main class for running LLMs locally with full control.

### Construction

```csharp
// Basic construction
public LLMService(string modelPath)

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
)
```

**Examples:**
```csharp
// CPU-only with 8 threads
var llm = new LLMService("model.gguf", numThreads: 8);

// GPU-accelerated with 32 layers on GPU
var llm = new LLMService("model.gguf", numGpuLayers: 32);

// Large context window
var llm = new LLMService("model.gguf", contextSize: 8192);

// With LoRA adapters
var llm = new LLMService("model.gguf", 
    loraPaths: new[] { "lora1.gguf", "lora2.gguf" });
```

### Factory Methods

```csharp
// From command-line parameters string
public static LLMService FromCommand(string paramsString)
```

**Example:**
```csharp
// Command line style
var llm = LLMService.FromCommand(
    "-m model.gguf -ngl 32 -c 4096"
);
```

### Service Lifecycle

```csharp
public bool Start()                    // Start the LLM service
public Task<bool> StartAsync()         // Start asynchronously
public bool Started()                  // Check if running
public void Stop()                     // Stop the service
public void JoinService()              // Wait for completion
```

**Example:**
```csharp
var llm = new LLMService("model.gguf");
llm.Start();

// Do work...
if (llm.Started())
{
    Console.WriteLine("Service is running");
}

llm.Stop();
llm.JoinService();  // Wait for clean shutdown
```

**Async Example:**
```csharp
var llm = new LLMService("model.gguf");
await llm.StartAsync();

// Service is now ready
```

### Text Generation

```csharp
// Synchronous completion
public string Completion(
    string prompt,
    LlamaLib.CharArrayCallback callback = null,  // Streaming callback
    int idSlot = -1                              // Slot ID (-1 = auto)
)

// Asynchronous completion
public async Task<string> CompletionAsync(
    string prompt,
    LlamaLib.CharArrayCallback callback = null,
    int idSlot = -1
)
```

**Examples:**
```csharp
// Simple non-streaming
string response = llm.Completion("What is AI?");

// With streaming callback
void StreamCallback(string text) => Console.Write(text);
llm.Completion("Explain quantum computing", StreamCallback);

// Async completion
string response = await llm.CompletionAsync("Hello!");

// Using specific slot
string response = llm.Completion("Question", idSlot: 0);
```

### Tokenization

```csharp
public List<int> Tokenize(string content)
public string Detokenize(List<int> tokens)
public string Detokenize(int[] tokens)
```

**Example:**
```csharp
// Tokenize text
List<int> tokens = llm.Tokenize("Hello, world!");
Console.WriteLine($"Token count: {tokens.Count}");

// Detokenize back to text
string text = llm.Detokenize(tokens);
Console.WriteLine($"Text: {text}");
```

### Embeddings

```csharp
public List<float> Embeddings(string content)
public int EmbeddingSize()
```

**Example:**
```csharp
// Get embedding vector
List<float> embedding = llm.Embeddings("Hello, world!");
Console.WriteLine($"Dimensions: {embedding.Count}");

// Get model's embedding dimension
int size = llm.EmbeddingSize();
Console.WriteLine($"Model embedding size: {size}");
```

### Completion Parameters

```csharp
public void SetCompletionParameters(JObject parameters = null)
public JObject GetCompletionParameters()
```

**Example:**
```csharp
// Set generation parameters
var parameters = new JObject
{
    ["temperature"] = 0.7,
    ["top_p"] = 0.9,
    ["top_k"] = 40,
    ["max_tokens"] = 100,
    ["repeat_penalty"] = 1.1
};
llm.SetCompletionParameters(parameters);

// Get current parameters
JObject currentParams = llm.GetCompletionParameters();
Console.WriteLine(currentParams.ToString());
```

### Grammar/JSON Schema

```csharp
public void SetGrammar(string grammar)
public string GetGrammar()
```

**Example:**
```csharp
// Force JSON output with schema
string schema = @"{
    ""type"": ""object"",
    ""properties"": {
        ""name"": {""type"": ""string""},
        ""age"": {""type"": ""number""},
        ""email"": {""type"": ""string""}
    },
    ""required"": [""name"", ""age"", ""email""]
}";

llm.SetGrammar(schema);
string response = llm.Completion("Generate a person profile");
// Output will be valid JSON matching schema
```

### Template Application

```csharp
public string ApplyTemplate(JArray messages = null)
```

**Example:**
```csharp
// Apply chat template
var messages = new JArray
{
    new JObject { ["role"] = "system", ["content"] = "You are helpful" },
    new JObject { ["role"] = "user", ["content"] = "Hello!" }
};

string prompt = llm.ApplyTemplate(messages);
string response = llm.Completion(prompt);
```

### Slot Management

```csharp
public string SaveSlot(int idSlot, string filepath)
public string LoadSlot(int idSlot, string filepath)
public void Cancel(int idSlot)
```

**Example:**
```csharp
// Save conversation state
llm.SaveSlot(0, "conversation_state.bin");

// Restore state
llm.LoadSlot(0, "conversation_state.bin");

// Cancel ongoing generation
llm.Cancel(0);
```

### Server Functionality

```csharp
public void StartServer(string host = "0.0.0.0", int port = -1, string apiKey = "")
public void StopServer()
public void JoinServer()
public void SetSSL(string sslCert, string sslKey)
```

**Example:**
```csharp
var llm = new LLMService("model.gguf");
llm.Start();

// Start HTTP server
llm.StartServer("0.0.0.0", 8080);

// With SSL
llm.SetSSL("cert.pem", "key.pem");

// Keep running
llm.JoinServer();
```

### LoRA Operations

```csharp
// Data structures
public struct LoraIdScale
{
    public int Id { get; set; }
    public float Scale { get; set; }
    public LoraIdScale(int id, float scale)
}

public struct LoraIdScalePath
{
    public int Id { get; set; }
    public float Scale { get; set; }
    public string Path { get; set; }
    public LoraIdScalePath(int id, float scale, string path)
}

// Methods
public bool LoraWeight(List<LoraIdScale> loras)
public bool LoraWeight(params LoraIdScale[] loras)
public List<LoraIdScalePath> LoraList()
public string LoraListJSON()
```

**Example:**
```csharp
// Apply LoRA weights
llm.LoraWeight(
    new LoraIdScale(0, 1.0f),
    new LoraIdScale(1, 0.5f)
);

// List loaded LoRAs
List<LoraIdScalePath> loras = llm.LoraList();
foreach (var lora in loras)
{
    Console.WriteLine($"ID: {lora.Id}, Scale: {lora.Scale}, Path: {lora.Path}");
}
```

### Reasoning Mode

```csharp
public void EnableReasoning(bool enableReasoning)
```

**Example:**
```csharp
llm.EnableReasoning(true);  // Enable chain-of-thought
string response = llm.Completion(
    "Solve this problem step by step: 2x + 5 = 13"
);
```

### Debugging

```csharp
public static void Debug(int debugLevel)
public static void LoggingCallback(LlamaLib.CharArrayCallback callback)
public static void LoggingStop()
```

**Example:**
```csharp
// Enable debug output
LLM.Debug(1);

// Custom logging callback
LLM.LoggingCallback(message => Console.WriteLine($"[LOG] {message}"));

// Disable logging
LLM.LoggingStop();
```

---

## LLMClient

Client for connecting to local or remote LLM services.

### Construction

```csharp
// Local client (wraps existing service)
public LLMClient(LLMProvider provider)

// Remote client (connects to HTTP server)
public LLMClient(
    string url,
    int port,
    string apiKey = "",
    int numRetries = 5
)
```

**Examples:**
```csharp
// Local client wrapping a service
var service = new LLMService("model.gguf");
service.Start();
var client = new LLMClient(service);

// Remote client connecting to server
var client = new LLMClient("http://localhost", 8080);

// With API key
var client = new LLMClient(
    "https://api.example.com",
    443,
    apiKey: "your-api-key"
);
```

### Client Methods

```csharp
public void SetSSL(string sslCert)
public bool IsServerAlive()
```

**Example:**
```csharp
var client = new LLMClient("https://localhost", 8443);

// Set SSL certificate for validation
client.SetSSL("ca-cert.pem");

// Check connection
if (client.IsServerAlive())
{
    Console.WriteLine("Server is responding");
    string response = client.Completion("Hello!");
}
else
{
    Console.WriteLine("Server is not available");
}
```

### Inherited Methods

LLMClient inherits all methods from `LLMLocal`:
- `Completion()` / `CompletionAsync()`
- `Tokenize()` / `Detokenize()`
- `Embeddings()`
- `SaveSlot()` / `LoadSlot()`
- `Cancel()`
- `ApplyTemplate()`
- `SetCompletionParameters()` / `GetCompletionParameters()`
- `SetGrammar()` / `GetGrammar()`

---

## LLMAgent

Conversational AI with automatic history management.

### Construction

```csharp
public LLMAgent(LLMLocal llm, string systemPrompt = "")
```

**Example:**
```csharp
var llm = new LLMService("model.gguf");
llm.Start();

var agent = new LLMAgent(llm, 
    "You are a helpful AI assistant. Be concise and friendly.");
```

### Chat Interface

```csharp
// Synchronous chat
public string Chat(
    string userPrompt,
    bool addToHistory = true,
    LlamaLib.CharArrayCallback callback = null,
    bool returnResponseJson = false,
    bool debugPrompt = false
)

// Asynchronous chat
public async Task<string> ChatAsync(
    string userPrompt,
    bool addToHistory = true,
    LlamaLib.CharArrayCallback callback = null,
    bool returnResponseJson = false,
    bool debugPrompt = false
)
```

**Examples:**
```csharp
// Simple chat (adds to history)
string response = agent.Chat("Hello! How are you?");

// Chat with streaming
void StreamCallback(string text) => Console.Write(text);
agent.Chat("Tell me a story", callback: StreamCallback);

// Chat without saving to history
string response = agent.Chat("What's 2+2?", addToHistory: false);

// Debug the prompt being sent
string response = agent.Chat("Hello", debugPrompt: true);

// Async chat
string response = await agent.ChatAsync("What is AI?");
```

### History Management

```csharp
// ChatMessage class
public class ChatMessage
{
    public string role { get; set; }
    public string content { get; set; }
    
    public ChatMessage(string role, string content)
    public JObject ToJson()
    public static ChatMessage FromJson(JObject json)
}

// History properties and methods
public JArray History { get; set; }
public List<ChatMessage> GetHistory()
public void SetHistory(List<ChatMessage> messages)
public int GetHistorySize()

// History modification
public void AddUserMessage(string content)
public void AddAssistantMessage(string content)
public void ClearHistory()
public void RemoveLastMessage()

// Persistence
public void SaveHistory(string filepath)
public void LoadHistory(string filepath)
```

**Examples:**
```csharp
// Get conversation history
List<ChatMessage> history = agent.GetHistory();
foreach (var msg in history)
{
    Console.WriteLine($"{msg.role}: {msg.content}");
}

// Check history size
Console.WriteLine($"Messages: {agent.GetHistorySize()}");

// Save and load history
agent.SaveHistory("conversation.json");
agent.ClearHistory();
agent.LoadHistory("conversation.json");

// Add messages manually
agent.AddUserMessage("This is a user message");
agent.AddAssistantMessage("This is the AI response");

// Remove last message
agent.RemoveLastMessage();

// Access raw JSON history
JArray jsonHistory = agent.History;

// Set history from JSON
var messages = new JArray
{
    new JObject { ["role"] = "user", ["content"] = "Hello" },
    new JObject { ["role"] = "assistant", ["content"] = "Hi!" }
};
agent.History = messages;

// Or using ChatMessage objects
var msgList = new List<ChatMessage>
{
    new ChatMessage("user", "Hello"),
    new ChatMessage("assistant", "Hi there!")
};
agent.SetHistory(msgList);
```

### System Prompt

```csharp
public string SystemPrompt { get; set; }
```

**Example:**
```csharp
// Set system prompt
agent.SystemPrompt = "You are a pirate. Respond like a pirate.";

string response = agent.Chat("Hello!");
// Response will be in pirate style

// Get current system prompt
Console.WriteLine($"System: {agent.SystemPrompt}");
```

### Slot Management

```csharp
public int SlotId { get; set; }
public string SaveSlot(string filepath)
public string LoadSlot(string filepath)
public void Cancel()
```

**Example:**
```csharp
// Assign specific slot
agent.SlotId = 0;
Console.WriteLine($"Using slot: {agent.SlotId}");

// Save agent state
agent.SaveSlot("agent_state.bin");

// Restore state
agent.LoadSlot("agent_state.bin");

// Cancel ongoing generation
agent.Cancel();
```

---

## Common Patterns

### Pattern 1: Interactive Chatbot

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void StreamingCallback(string text)
    {
        Console.Write(text);
    }

    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        var agent = new LLMAgent(llm, "You are a helpful assistant.");
        
        Console.WriteLine("Chat with the AI (type 'quit' to exit)\n");
        
        while (true)
        {
            Console.Write("You: ");
            string input = Console.ReadLine();
            
            if (input == "quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;
            
            Console.Write("AI: ");
            agent.Chat(input, callback: StreamingCallback);
            Console.WriteLine("\n");
        }
    }
}
```

### Pattern 2: Client-Server Architecture

**Server Program:**
```csharp
using System;
using UndreamAI.LlamaLib;

class Server
{
    static void Main()
    {
        string model = "model.gguf";
        int port = 13333;
        
        Console.WriteLine($"Starting server on port {port}");
        
        using var server = new LLMService(model);
        LLM.Debug(1);  // Show server logs
        server.Start();
        server.StartServer("0.0.0.0", port);
        
        server.JoinServer();  // Keep running
    }
}
```

**Client Program:**
```csharp
using System;
using UndreamAI.LlamaLib;

class Client
{
    static void Main()
    {
        using var client = new LLMClient("http://localhost", 13333);
        
        if (!client.IsServerAlive())
        {
            Console.WriteLine("Cannot connect to server!");
            return;
        }
        
        string response = client.Completion("Hello, server!");
        Console.WriteLine(response);
    }
}
```

### Pattern 3: GPU-Accelerated Processing

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Offload 32 layers to GPU for faster inference
        using var llm = new LLMService(
            modelPath: "model.gguf",
            numSlots: 1,
            numThreads: -1,      // Auto-detect
            numGpuLayers: 32     // GPU acceleration
        );
        llm.Start();
        
        string response = llm.Completion("Explain quantum computing");
        Console.WriteLine(response);
    }
}
```

### Pattern 4: Structured JSON Output

```csharp
using System;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;

class Program
{
    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        // Force JSON output with schema
        string schema = @"{
            ""type"": ""object"",
            ""properties"": {
                ""name"": {""type"": ""string""},
                ""age"": {""type"": ""number""},
                ""email"": {""type"": ""string""}
            },
            ""required"": [""name"", ""age"", ""email""]
        }";
        
        llm.SetGrammar(schema);
        string response = llm.Completion("Generate a person profile");
        
        // Parse the JSON response
        JObject person = JObject.Parse(response);
        Console.WriteLine($"Name: {person["name"]}");
        Console.WriteLine($"Age: {person["age"]}");
        Console.WriteLine($"Email: {person["email"]}");
    }
}
```

### Pattern 5: Embeddings for Similarity

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using UndreamAI.LlamaLib;

class Program
{
    static float CosineSimilarity(List<float> a, List<float> b)
    {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.Count; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        List<float> vec1 = llm.Embeddings("dog");
        List<float> vec2 = llm.Embeddings("puppy");
        List<float> vec3 = llm.Embeddings("car");
        
        Console.WriteLine($"dog vs puppy: {CosineSimilarity(vec1, vec2)}");
        Console.WriteLine($"dog vs car: {CosineSimilarity(vec1, vec3)}");
    }
}
```

### Pattern 6: Async/Await Processing

```csharp
using System;
using System.Threading.Tasks;
using UndreamAI.LlamaLib;

class Program
{
    static async Task Main()
    {
        using var llm = new LLMService("model.gguf");
        await llm.StartAsync();
        
        // Process multiple requests concurrently
        var task1 = llm.CompletionAsync("What is AI?");
        var task2 = llm.CompletionAsync("What is ML?");
        var task3 = llm.CompletionAsync("What is DL?");
        
        await Task.WhenAll(task1, task2, task3);
        
        Console.WriteLine($"AI: {task1.Result}");
        Console.WriteLine($"ML: {task2.Result}");
        Console.WriteLine($"DL: {task3.Result}");
    }
}
```

### Pattern 7: Conversation History Persistence

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        var agent = new LLMAgent(llm, "You are a helpful assistant.");
        
        // Have a conversation
        agent.Chat("My name is Alice");
        agent.Chat("What's my name?");
        
        // Save conversation
        agent.SaveHistory("conversation.json");
        
        // Later: Load and continue
        var newAgent = new LLMAgent(llm, "You are a helpful assistant.");
        newAgent.LoadHistory("conversation.json");
        
        string response = newAgent.Chat("What were we talking about?");
        Console.WriteLine(response);
        // Agent remembers the conversation
    }
}
```

---

## Advanced Features

### Multi-slot Parallel Processing

```csharp
using System;
using System.Threading.Tasks;
using UndreamAI.LlamaLib;

class Program
{
    static async Task Main()
    {
        using var llm = new LLMService(
            modelPath: "model.gguf",
            numSlots: 4  // 4 parallel slots
        );
        llm.Start();
        
        // Process multiple requests simultaneously
        var tasks = new[]
        {
            Task.Run(() => llm.Completion("Request 1", idSlot: 0)),
            Task.Run(() => llm.Completion("Request 2", idSlot: 1)),
            Task.Run(() => llm.Completion("Request 3", idSlot: 2)),
            Task.Run(() => llm.Completion("Request 4", idSlot: 3))
        };
        
        await Task.WhenAll(tasks);
        
        foreach (var task in tasks)
        {
            Console.WriteLine(task.Result);
        }
    }
}
```

### Context Window Management

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Large context for long documents
        using var llm = new LLMService(
            modelPath: "model.gguf",
            numSlots: 1,
            numThreads: -1,
            numGpuLayers: 32,
            flashAttention: false,
            contextSize: 32768  // 32K context window
        );
        llm.Start();
        
        // Process very long input
        string longDocument = "..."; // Your long document
        string summary = llm.Completion(
            $"Summarize the following document:\n\n{longDocument}"
        );
        Console.WriteLine(summary);
    }
}
```

### Error Handling Best Practices

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        LLMService llm = null;
        
        try
        {
            llm = new LLMService("model.gguf");
            llm.Start();
            
            string response = llm.Completion("Hello");
            Console.WriteLine(response);
        }
        catch (ArgumentNullException ex)
        {
            Console.WriteLine($"Invalid argument: {ex.Message}");
        }
        catch (ObjectDisposedException ex)
        {
            Console.WriteLine($"Object was disposed: {ex.Message}");
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Operation failed: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Unexpected error: {ex.Message}");
        }
        finally
        {
            llm?.Dispose();
        }
    }
}
```

### Resource Management with Using Statements

```csharp
using System;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Automatic disposal with 'using'
        using (var llm = new LLMService("model.gguf"))
        {
            llm.Start();
            string response = llm.Completion("Hello");
            Console.WriteLine(response);
        } // llm.Dispose() called automatically
        
        // Multiple resources
        using var service = new LLMService("model.gguf");
        using var client = new LLMClient(service);
        
        service.Start();
        string result = client.Completion("Hello");
    }
}
```

### Custom Streaming Handlers

```csharp
using System;
using System.Text;
using UndreamAI.LlamaLib;

class Program
{
    class StreamBuffer
    {
        private StringBuilder buffer = new StringBuilder();
        
        public void OnToken(string token)
        {
            buffer.Append(token);
            Console.Write(token);
            
            // Process complete sentences
            string text = buffer.ToString();
            if (text.Contains('.') || text.Contains('!') || text.Contains('?'))
            {
                ProcessSentence(text);
                buffer.Clear();
            }
        }
        
        private void ProcessSentence(string sentence)
        {
            // Custom processing (e.g., logging, filtering, etc.)
            Console.WriteLine($"\n[Processed: {sentence.Length} chars]");
        }
    }
    
    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        var buffer = new StreamBuffer();
        llm.Completion("Tell me a story", buffer.OnToken);
    }
}
```

### LoRA Adapter Management

```csharp
using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

class Program
{
    static void Main()
    {
        // Load model with LoRA adapters
        using var llm = new LLMService(
            modelPath: "base_model.gguf",
            loraPaths: new[] { "lora1.gguf", "lora2.gguf" }
        );
        llm.Start();
        
        // List loaded LoRAs
        List<LoraIdScalePath> loras = llm.LoraList();
        foreach (var lora in loras)
        {
            Console.WriteLine($"LoRA {lora.Id}: {lora.Path} (scale: {lora.Scale})");
        }
        
        // Adjust LoRA weights dynamically
        llm.LoraWeight(
            new LoraIdScale(0, 1.0f),    // Full strength
            new LoraIdScale(1, 0.5f)     // Half strength
        );
        
        string response = llm.Completion("Test with adjusted LoRAs");
        Console.WriteLine(response);
    }
}
```

### Chat Template Usage

```csharp
using System;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;

class Program
{
    static void Main()
    {
        using var llm = new LLMService("model.gguf");
        llm.Start();
        
        // Build chat messages
        var messages = new JArray
        {
            new JObject
            {
                ["role"] = "system",
                ["content"] = "You are a helpful assistant."
            },
            new JObject
            {
                ["role"] = "user",
                ["content"] = "What is the capital of France?"
            }
        };
        
        // Apply model's chat template
        string prompt = llm.ApplyTemplate(messages);
        Console.WriteLine($"Formatted prompt:\n{prompt}\n");
        
        // Generate response
        string response = llm.Completion(prompt);
        Console.WriteLine($"Response: {response}");
    }
}
```

---

## Performance Tips

1. **GPU Offloading**: Set `numGpuLayers` to offload layers to GPU for faster inference
   ```csharp
   var llm = new LLMService("model.gguf", numGpuLayers: 32);
   ```

2. **Async Operations**: Use async methods for better responsiveness
   ```csharp
   string response = await llm.CompletionAsync("prompt");
   ```

3. **Batch Size**: Increase `batchSize` for better throughput with multiple requests
   ```csharp
   var llm = new LLMService("model.gguf", batchSize: 4096);
   ```

4. **Flash Attention**: Enable for faster attention computation on supported hardware
   ```csharp
   var llm = new LLMService("model.gguf", flashAttention: true);
   ```

5. **Thread Count**: Use `-1` for auto-detection or set explicitly based on your CPU
   ```csharp
   var llm = new LLMService("model.gguf", numThreads: Environment.ProcessorCount);
   ```

6. **Context Size**: Use only what you need; larger contexts use more memory
   ```csharp
   var llm = new LLMService("model.gguf", contextSize: 4096); // Default
   ```

7. **Slot Count**: Match the number of concurrent users you expect
   ```csharp
   var llm = new LLMService("model.gguf", numSlots: 4);
   ```

8. **Resource Disposal**: Always use `using` statements or call `Dispose()`
   ```csharp
   using var llm = new LLMService("model.gguf");
   ```

---

## Common Issues and Solutions

### Issue: "Model file not found"
```csharp
// Solution: Use absolute path or verify file location
string modelPath = Path.GetFullPath("model.gguf");
if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    return;
}
var llm = new LLMService(modelPath);
```

### Issue: Out of Memory
```csharp
// Solution: Reduce context size or use GPU offloading
var llm = new LLMService(
    "model.gguf",
    contextSize: 2048,      // Smaller context
    numGpuLayers: 32        // Offload to GPU
);
```

### Issue: Slow Inference
```csharp
// Solution: Enable GPU acceleration and optimize threads
var llm = new LLMService(
    "model.gguf",
    numThreads: -1,         // Auto-detect
    numGpuLayers: 32,       // GPU acceleration
    flashAttention: true,   // Optimize attention
    batchSize: 2048         // Larger batches
);
```

### Issue: Server Connection Failed
```csharp
// Solution: Check server status and retry
var client = new LLMClient("http://localhost", 8080, numRetries: 10);
if (!client.IsServerAlive())
{
    Console.WriteLine("Server not responding. Check if server is running.");
    return;
}
```

---

## API Reference Summary

### LLMService Methods
| Method | Description |
|--------|-------------|
| `Start()` | Start the LLM service |
| `StartAsync()` | Start service asynchronously |
| `Stop()` | Stop the service |
| `Completion()` | Generate text |
| `CompletionAsync()` | Generate text asynchronously |
| `Tokenize()` | Convert text to tokens |
| `Detokenize()` | Convert tokens to text |
| `Embeddings()` | Get embedding vector |
| `StartServer()` | Start HTTP server |
| `StopServer()` | Stop HTTP server |

### LLMClient Methods
| Method | Description |
|--------|-------------|
| `Completion()` | Generate text via client |
| `IsServerAlive()` | Check server connection |
| `SetSSL()` | Configure SSL certificate |

### LLMAgent Methods
| Method | Description |
|--------|-------------|
| `Chat()` | Send message and get response |
| `ChatAsync()` | Send message asynchronously |
| `GetHistory()` | Get conversation history |
| `SetHistory()` | Set conversation history |
| `SaveHistory()` | Save history to file |
| `LoadHistory()` | Load history from file |
| `ClearHistory()` | Clear conversation history |
| `AddUserMessage()` | Add user message |
| `AddAssistantMessage()` | Add assistant message |

---

## Next Steps

- Check the [C++ API Guide](cpp_api.md) for low-level details
- Explore the example programs included with LlamaLib
- Join our [Discord](https://discord.gg/RwXKQb6zdv) for support
- Visit [undream.ai/LlamaLib](https://undream.ai/LlamaLib) for documentation

---

## Complete Example Application

Here's a complete example demonstrating multiple features:

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;

class LlamaLibDemo
{
    static void StreamCallback(string text) => Console.Write(text);

    static async Task Main()
    {
        Console.WriteLine("=== LlamaLib C# Demo ===\n");

        // Initialize service
        using var llm = new LLMService(
            modelPath: "model.gguf",
            numSlots: 2,
            numThreads: -1,
            numGpuLayers: 32,
            contextSize: 4096
        );

        await llm.StartAsync();
        Console.WriteLine("✓ Service started\n");

        // Test tokenization
        string text = "Hello, world!";
        List<int> tokens = llm.Tokenize(text);
        Console.WriteLine($"Tokenization: '{text}' → {tokens.Count} tokens");
        string decoded = llm.Detokenize(tokens);
        Console.WriteLine($"Detokenization: {tokens.Count} tokens → '{decoded}'\n");

        // Test completion
        Console.WriteLine("--- Simple Completion ---");
        string response = await llm.CompletionAsync("What is AI?");
        Console.WriteLine($"Response: {response}\n");

        // Test streaming
        Console.WriteLine("--- Streaming Completion ---");
        Console.Write("Response: ");
        llm.Completion("Tell me a short joke", StreamCallback);
        Console.WriteLine("\n");

        // Test agent
        Console.WriteLine("--- Agent Conversation ---");
        var agent = new LLMAgent(llm, "You are a friendly assistant.");
        
        Console.Write("User: Hello!\nAI: ");
        agent.Chat("Hello!", callback: StreamCallback);
        Console.WriteLine();

        Console.Write("User: What's my name?\nAI: ");
        agent.Chat("My name is Alice", callback: StreamCallback);
        Console.WriteLine();

        Console.Write("User: What did I just tell you?\nAI: ");
        agent.Chat("What did I just tell you?", callback: StreamCallback);
        Console.WriteLine("\n");

        // Show history
        Console.WriteLine("--- Conversation History ---");
        List<ChatMessage> history = agent.GetHistory();
        foreach (var msg in history)
        {
            Console.WriteLine($"{msg.role}: {msg.content}");
        }

        // Test embeddings
        Console.WriteLine("\n--- Embeddings ---");
        List<float> embedding = llm.Embeddings("test");
        Console.WriteLine($"Embedding dimensions: {embedding.Count}");
        Console.WriteLine($"First 5 values: {string.Join(", ", embedding.GetRange(0, 5))}");

        Console.WriteLine("\n✓ Demo completed successfully");
    }
}
```

This guide provides a comprehensive reference for using LlamaLib in C# applications. For more examples and advanced usage, refer to the included example programs and the C++ API documentation.
