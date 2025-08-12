# LLM Classes C# API Documentation

## Class Overview

This documentation covers the public methods for three main LLM classes:

- **LLMService**: Concrete implementation of LLMProvider with HTTP server functionality, parameter parsing, and integration with llama.cpp backend
- **LLMClient**: Client for accessing LLM functionality locally or remotely with unified interface for both local LLMProvider instances and remote LLM services via HTTP
- **LLMAgent**: High-level conversational agent that manages chat history and applies chat template formatting

## Method Support Matrix

| Method | LLMService | LLMClient | LLMAgent |
|--------|------------|-----------|----------|
| **Construction & Setup** | | | |
| Constructor (basic) | ✅ | ✅ | ✅ |
| Constructor (remote) | ❌ | ✅ | ❌ |
| Factory methods (FromCommand) | ✅ | ❌ | ❌ |
| **Core LLM Operations** | | | |
| Tokenize() | ✅ | ✅ | ✅ |
| Detokenize() | ✅ | ✅ | ✅ |
| Embeddings() | ✅ | ✅ | ✅ |
| Completion() | ✅ | ✅ | ✅ |
| CompletionAsync() | ✅ | ✅ | ✅ |
| GetTemplate() | ✅ | ✅ | ✅ |
| ApplyTemplate() | ✅ | ✅ | ✅ |
| **Configuration** | | | |
| SetCompletionParameters() | ✅ | ✅ | ✅ |
| GetCompletionParameters() | ✅ | ✅ | ✅ |
| SetGrammar() | ✅ | ✅ | ✅ |
| GetGrammar() | ✅ | ✅ | ✅ |
| SetTemplate() | ✅ | ❌ | ❌ |
| **Slot Management** | | | |
| SaveSlot() | ✅ | ✅ | ✅ |
| LoadSlot() | ✅ | ✅ | ✅ |
| Cancel() | ✅ | ✅ | ✅ |
| **Service Management** | | | |
| Start() | ✅ | ❌ | ❌ |
| StartAsync() | ✅ | ❌ | ❌ |
| Started() | ✅ | ❌ | ❌ |
| Stop() | ✅ | ❌ | ❌ |
| StartServer() | ✅ | ❌ | ❌ |
| StopServer() | ✅ | ❌ | ❌ |
| JoinService() | ✅ | ❌ | ❌ |
| JoinServer() | ✅ | ❌ | ❌ |
| SetSSL() | ✅ | ❌ | ❌ |
| **LoRA Operations** | | | |
| LoraWeight() | ✅ | ❌ | ❌ |
| LoraList() | ✅ | ❌ | ❌ |
| **Debugging & Utilities** | | | |
| EmbeddingSize() | ✅ | ❌ | ❌ |
| Static Debug methods | ✅ | ✅ | ✅ |
| **Agent-Specific Methods** | | | |
| Chat() / ChatAsync() | ❌ | ❌ | ✅ |
| SlotId property | ❌ | ❌ | ✅ |
| Role properties | ❌ | ❌ | ✅ |
| History management | ❌ | ❌ | ✅ |

---

## LLMService

Concrete implementation of LLMProvider with HTTP server functionality, parameter parsing, and integration with llama.cpp backend. This class provides a full-featured LLM service with HTTP server, parameter configuration, and backend integration optimized for .NET applications with async support and automatic resource management.

### Construction & Factory Methods

```csharp
// Parameterized constructor
public LLMService(string modelPath, int numSlots = 1, int numThreads = -1, 
                  int numGpuLayers = 0, bool flashAttention = false, int contextSize = 4096, 
                  int batchSize = 2048, bool embeddingOnly = false, string[] loraPaths = null)

// Factory method for command line string
public static LLMService FromCommand(string paramsString)
```

**Description**: Creates LLMService instances with parameter configuration including GPU layer offloading, context size, and LoRA adapter paths. Factory method allows creation from command line arguments compatible with llama.cpp server parameters.

### Core LLM Operations

```csharp
// Tokenize text into token IDs (inherited from LLM)
public List<int> Tokenize(string content)

// Convert token IDs back to text (inherited from LLM)
public string Detokenize(List<int> tokens)
public string Detokenize(int[] tokens)

// Generate embeddings for text (inherited from LLM)
public List<float> Embeddings(string content)

// Generate text completion (inherited from LLM)
public string Completion(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1)
public async Task<string> CompletionAsync(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1)

// Get/apply chat template (inherited from LLM)
public string GetTemplate()
public string ApplyTemplate(JArray messages = null)
```

**Description**: Core language model operations with .NET collections (List<int>, List<float>) and async patterns for non-blocking operations, plus JSON.NET integration for structured data.

### Service Management

```csharp
// Start the LLM service
public bool Start()
public async Task<bool> StartAsync()

// Check if service is running
public bool Started()

// Stop the LLM service
public void Stop()

// Start HTTP server
public void StartServer(string host = "0.0.0.0", int port = -1, string apiKey = "")

// Stop HTTP server
public void StopServer()

// Wait for service/server thread completion
public void JoinService()
public void JoinServer()
```

**Description**: Service lifecycle management with async support for non-blocking startup operations, HTTP server functionality with configurable host/port/API key, and thread synchronization methods.

### Configuration

```csharp
// Set/get completion parameters (inherited from LLM)
public void SetCompletionParameters(JObject parameters = null)
public JObject GetCompletionParameters()

// Set/get grammar for constrained generation (inherited from LLM)
public void SetGrammar(string grammar)
public string GetGrammar()

// Set chat template
public void SetTemplate(string template)

// Configure SSL certificates
public void SetSSL(string sslCert, string sslKey)
```

**Description**: Configuration methods using JSON.NET objects for completion parameters, grammar constraints in GBNF format or JSON schema, chat template specification, and SSL certificate paths for secure HTTPS connections.

### LoRA Operations

```csharp
// Configure LoRA weights
public bool LoraWeight(List<LoraIdScale> loras)
public bool LoraWeight(params LoraIdScale[] loras)

// List available LoRA adapters
public List<LoraIdScalePath> LoraList()
```

**Description**: LoRA (Low-Rank Adaptation) management with strongly-typed structures, supporting both collection and parameter array syntax for weight configuration, returning success status and available adapter listings.

### Slot Management

```csharp
// Save/load slot state (inherited from LLMLocal)
public string SaveSlot(int idSlot, string filepath)
public string LoadSlot(int idSlot, string filepath)

// Cancel running request (inherited from LLMLocal)
public void Cancel(int idSlot)
```

**Description**: Slot operations for concurrent request processing with state persistence to files and request cancellation for specific processing slots.

### Utilities

```csharp
// Get embedding vector dimensions
public int EmbeddingSize()

// Static debugging methods
public static void Debug(int debugLevel)
public static void LoggingCallback(LlamaLib.CharArrayCallback callback)
public static void LoggingStop()
```

**Description**: Utility methods for embeddings dimension information and global debugging configuration where debugLevel controls verbosity and callbacks receive log messages.

---

## LLMClient

Client for accessing LLM functionality locally or remotely with unified interface for both local LLMProvider instances and remote LLM services via HTTP. Provides seamless access to LLM functionality whether connecting to local providers or remote HTTP endpoints, with automatic resource management and exception handling.

### Construction

```csharp
// Constructor for local LLM access
public LLMClient(LLMProvider provider)

// Constructor for remote LLM access
public LLMClient(string url, int port, string apiKey = "")
```

**Description**: Creates clients for local providers or remote HTTP endpoints with automatic resource management, disposing of resources properly when connecting fails or objects are disposed.

### Core LLM Operations

```csharp
// All core LLM operations inherited from LLMLocal
// Including: Tokenize, Detokenize, Embeddings, Completion, GetTemplate, ApplyTemplate
```

**Description**: Full LLM functionality inherited from base classes, working transparently with local or remote backends using the same API surface regardless of connection type.

---

## LLMAgent

High-level conversational agent that manages chat history and applies chat template formatting. Wraps an LLMLocal instance with conversation management, providing automatic history tracking, role-based messaging, and persistent conversation state with strongly-typed ChatMessage objects.

### Construction

```csharp
// Constructor for conversational agent
public LLMAgent(LLMLocal _llm, string _systemPrompt = "", string _userRole = "user", string _assistantRole = "assistant")
```

**Description**: Creates a conversational agent that wraps an LLMLocal instance with chat history management, configurable system prompt for context, and customizable role identifiers for conversation participants.

### Core Chat Functionality

```csharp
// Conduct a chat interaction
public string Chat(string userPrompt, bool addToHistory = true, LlamaLib.CharArrayCallback callback = null, bool returnResponseJson = false)
public async Task<string> ChatAsync(string userPrompt, bool addToHistory = true, LlamaLib.CharArrayCallback callback = null, bool returnResponseJson = false)
```

**Description**: Main chat methods with sync/async support for conversational interactions, automatic history management, optional streaming callbacks, and JSON response format control.

### Properties

```csharp
// Slot management
public int SlotId { get; set; }

// Role configuration
public string UserRole { get; set; }
public string AssistantRole { get; set; }
public string SystemPrompt { get; set; }

// History access
public JArray History { get; set; }
```

**Description**: Properties for accessing and configuring agent state including slot assignment for processing isolation, role identifiers for message attribution, system prompt for context, and direct history access as JSON array.

### History Management

```csharp
// Get/set history as strongly-typed collections
public List<ChatMessage> GetHistory()
public void SetHistory(List<ChatMessage> messages)

// Clear all conversation history
public void ClearHistory()

// Add messages to conversation history
public void AddMessage(string role, string content)
public void AddUserMessage(string content)
public void AddAssistantMessage(string content)
public void AddMessage(ChatMessage message)

// Remove the last message from history
public void RemoveLastMessage()

// Save/load conversation history to/from file
public void SaveHistory(string filepath)
public void LoadHistory(string filepath)

// Get number of messages in history
public int GetHistorySize()
```

**Description**: Comprehensive history management with strongly-typed ChatMessage objects for type safety, role-specific convenience methods for common operations, file-based persistence as JSON, history manipulation methods, and size tracking.

### Slot-Aware Operations

```csharp
// Agent-specific completion using assigned slot
public string Completion(string prompt, LlamaLib.CharArrayCallback callback = null)
public async Task<string> CompletionAsync(string prompt, LlamaLib.CharArrayCallback callback = null)

// Slot state management using agent's slot
public string SaveSlot(string filepath)
public string LoadSlot(string filepath)

// Cancel agent's current request
public void Cancel()
```

**Description**: Slot-aware operations that automatically use the agent's assigned slot, hiding complexity from users by not requiring slot ID parameters, with async support for non-blocking operations and automatic state management.