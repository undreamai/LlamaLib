# LLM Classes API Documentation

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
| Factory methods (from_params/from_command) | ✅ | ❌ | ❌ |
| **Core LLM Operations** | | | |
| tokenize() | ✅ | ✅ | ✅ |
| detokenize() | ✅ | ✅ | ✅ |
| embeddings() | ✅ | ✅ | ✅ |
| completion() | ✅ | ✅ | ✅ |
| completion_json() | ✅ | ✅ | ✅ |
| get_template() | ✅ | ✅ | ✅ |
| apply_template() | ✅ | ✅ | ✅ |
| **Configuration** | | | |
| set_completion_params() | ✅ | ✅ | ✅ |
| get_completion_params() | ✅ | ✅ | ✅ |
| set_grammar() | ✅ | ✅ | ✅ |
| get_grammar() | ✅ | ✅ | ✅ |
| set_template() | ✅ | ❌ | ❌ |
| **Slot Management** | | | |
| get_next_available_slot() | ✅ | ✅ | ✅ |
| save_slot() | ✅ | ✅ | ✅ |
| load_slot() | ✅ | ✅ | ✅ |
| cancel() | ✅ | ✅ | ✅ |
| **Service Management** | | | |
| start() | ✅ | ❌ | ❌ |
| started() | ✅ | ❌ | ❌ |
| stop() | ✅ | ❌ | ❌ |
| start_server() | ✅ | ❌ | ❌ |
| stop_server() | ✅ | ❌ | ❌ |
| join_service() | ✅ | ❌ | ❌ |
| join_server() | ✅ | ❌ | ❌ |
| set_SSL() | ✅ | ✅ | ❌ |
| **LoRA Operations** | | | |
| lora_weight() | ✅ | ❌ | ❌ |
| lora_list() | ✅ | ❌ | ❌ |
| **Debugging & Utilities** | | | |
| debug() | ✅ | ❌ | ❌ |
| logging_callback() | ✅ | ❌ | ❌ |
| embedding_size() | ✅ | ❌ | ❌ |
| **Agent-Specific Methods** | | | |
| chat() | ❌ | ❌ | ✅ |
| get_slot() / set_slot() | ❌ | ❌ | ✅ |
| Role management | ❌ | ❌ | ✅ |
| History management | ❌ | ❌ | ✅ |
| **Client-Specific** | | | |
| is_remote() | ❌ | ✅ | ❌ |

---

<div class="api-tabs">
  <div class="tab-buttons">
    <button class="tab-button active" onclick="showTab('cpp')">C++</button>
    <button class="tab-button" onclick="showTab('csharp')">C#</button>
  </div>

<div id="cpp-content" class="tab-content active">

## C++ API

### LLMService

#### Construction & Factory Methods

```cpp
// Default constructor - creates uninitialized service
LLMService();

// Parameterized constructor
LLMService(const std::string &model_path, int num_slots = 1, int num_threads = -1, 
           int num_GPU_layers = 0, bool flash_attention = false, int context_size = 4096, 
           int batch_size = 2048, bool embedding_only = false, 
           const std::vector<std::string> &lora_paths = {});

// Factory method for JSON parameters
static LLMService *from_params(const json &params_json);

// Factory method for command line string
static LLMService *from_command(const std::string &command);

// Factory method for argc/argv
static LLMService *from_command(int argc, char **argv);
```

**Description**: Creates LLMService instances with various initialization methods. Factory methods allow creation from structured JSON parameters or command line arguments compatible with llama.cpp server.

#### Core LLM Operations

```cpp
// Tokenize text into token IDs
std::vector<int> tokenize(const std::string &query);

// Convert token IDs back to text
std::string detokenize(const std::vector<int32_t> &tokens);

// Generate embeddings for text
std::vector<float> embeddings(const std::string &query);

// Generate text completion
std::string completion(const std::string &prompt, CharArrayFn callback = nullptr, 
                      int id_slot = -1, bool return_response_json = false);

// Generate completion from JSON data
std::string completion_json(const json &data, CharArrayFn callback = nullptr, 
                           bool callbackWithJSON = true);

// Get chat template
std::string get_template();

// Apply template to messages
std::string apply_template(const json &messages);
```

**Description**: Core language model operations for tokenization, text generation, embeddings, and template processing.

#### Service Management

```cpp
// Start the LLM service
void start();

// Check if service is running
bool started();

// Stop the LLM service
void stop();

// Start HTTP server
void start_server(const std::string &host = "0.0.0.0", int port = -1, 
                  const std::string &API_key = "");

// Stop HTTP server
void stop_server();

// Wait for service thread completion
void join_service();

// Wait for server thread completion
void join_server();
```

**Description**: Methods for managing the LLM service lifecycle and HTTP server functionality.

#### Configuration

```cpp
// Set completion parameters
void set_completion_params(json completion_params_);

// Get current completion parameters
std::string get_completion_params();

// Set grammar for constrained generation
void set_grammar(std::string grammar_);

// Get current grammar specification
std::string get_grammar();

// Set chat template
void set_template(std::string chat_template);

// Configure SSL certificates
void set_SSL(const std::string &SSL_cert, const std::string &SSL_key);
```

**Description**: Configuration methods for completion parameters, grammar constraints, chat templates, and SSL setup.

#### LoRA Operations

```cpp
// Configure LoRA weights
bool lora_weight(const std::vector<LoraIdScale> &loras);

// List available LoRA adapters
std::vector<LoraIdScalePath> lora_list();
```

**Description**: Methods for managing Low-Rank Adaptation (LoRA) layers including weight configuration and adapter listing.

#### Slot Management

```cpp
// Get available processing slot
int get_next_available_slot();

// Save slot state to file
std::string save_slot(int id_slot, const std::string &filepath);

// Load slot state from file
std::string load_slot(int id_slot, const std::string &filepath);

// Cancel running request
void cancel(int id_slot);
```

**Description**: Slot management for concurrent request processing, state persistence, and request cancellation.

#### Debugging & Utilities

```cpp
// Set debug level
void debug(int debug_level);

// Set logging callback function
void logging_callback(CharArrayFn callback);

// Get embedding vector dimensions
int embedding_size();
```

**Description**: Debugging, logging, and utility methods for service monitoring and configuration.

---

### LLMClient

#### Construction

```cpp
// Constructor for local LLM access
LLMClient(LLMProvider *llm);

// Constructor for remote LLM access
LLMClient(const std::string &url, const int port, const std::string &API_key = "");
```

**Description**: Creates clients for either local LLMProvider instances or remote LLM services via HTTP.

#### Core LLM Operations

```cpp
// Tokenize text (inherited from LLMLocal)
std::vector<int> tokenize(const std::string &query);

// Convert tokens to text (inherited from LLMLocal)
std::string detokenize(const std::vector<int32_t> &tokens);

// Generate embeddings (inherited from LLMLocal)
std::vector<float> embeddings(const std::string &query);

// Generate completion (inherited from LLMLocal)
std::string completion_json(const json &data, CharArrayFn callback = nullptr, 
                           bool callbackWithJSON = true);

// Get template (inherited from LLMLocal)
std::string get_template();

// Apply template to messages (inherited from LLMLocal)
std::string apply_template(const json &messages);
```

**Description**: Unified interface for LLM operations that works with both local and remote backends.

#### Slot Management

```cpp
// Get available processing slot (inherited from LLMLocal)
int get_next_available_slot();

// Cancel running request (inherited from LLMLocal)
void cancel(int id_slot);
```

**Description**: Slot management methods that work across local and remote connections.

#### Client-Specific Methods

```cpp
// Configure SSL certificate for remote connections
void set_SSL(const char *SSL_cert);

// Check if this is a remote client
bool is_remote() const;
```

**Description**: Client-specific configuration and utility methods for SSL setup and connection type detection.

---

### LLMAgent

#### Construction

```cpp
// Constructor for conversational agent
LLMAgent(LLMLocal *llm, const std::string &system_prompt = "", 
         const std::string &user_role = "user", const std::string &assistant_role = "assistant");
```

**Description**: Creates a conversational agent that manages chat history and applies template formatting around an LLMLocal instance.

#### Core Chat Functionality

```cpp
// Conduct a chat interaction
std::string chat(const std::string &user_prompt, bool add_to_history = true, 
                CharArrayFn callback = nullptr, bool return_response_json = false);
```

**Description**: Main chat method that processes user input, applies conversation context, generates a response, and optionally updates conversation history.

#### Slot Management

```cpp
// Get current processing slot ID
inline int get_slot();

// Set processing slot ID
void set_slot(int id_slot);

// Save agent's slot state
std::string save_slot(const std::string &filepath);

// Load agent's slot state
std::string load_slot(const std::string &filepath);

// Cancel agent's current request
void cancel();
```

**Description**: Slot-aware methods that automatically use the agent's assigned slot for operations.

#### Role Configuration

```cpp
// Set/get user role identifier
void set_user_role(const std::string &user_role_);
std::string get_user_role() const;

// Set/get assistant role identifier
void set_assistant_role(const std::string &assistant_role_);
std::string get_assistant_role() const;

// Set/get system prompt
void set_system_prompt(const std::string &system_prompt_);
std::string get_system_prompt() const;
```

**Description**: Configuration methods for chat roles and system prompts that define conversation context.

#### History Management

```cpp
// Set/get conversation history
void set_history(const json &history_);
json get_history() const;

// Add messages to conversation history
void add_message(const std::string &role, const std::string &content);
void add_user_message(const std::string &content);
void add_assistant_message(const std::string &content);

// Clear all conversation history
void clear_history();

// Remove the last message from history
void remove_last_message();

// Save/load conversation history to/from file
void save_history(const std::string &filepath) const;
void load_history(const std::string &filepath);

// Get number of messages in history
size_t get_history_size() const;
```

**Description**: Comprehensive history management for persistent conversations, including file I/O and message manipulation.

</div>

<div id="csharp-content" class="tab-content">

## C# API

### LLMService

#### Construction & Factory Methods

```csharp
// Parameterized constructor
public LLMService(string modelPath, int numSlots = 1, int numThreads = -1, 
                  int numGpuLayers = 0, bool flashAttention = false, int contextSize = 4096, 
                  int batchSize = 2048, bool embeddingOnly = false, string[] loraPaths = null)

// Factory method for command line string
public static LLMService FromCommand(string paramsString)
```

**Description**: Creates LLMService instances with parameter configuration or from command line arguments compatible with llama.cpp server.

#### Core LLM Operations

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

**Description**: Core language model operations with support for async patterns and JSON.NET integration.

#### Service Management

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

**Description**: Service lifecycle management with async support for non-blocking operations.

#### Configuration

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

**Description**: Configuration methods using JSON.NET objects for parameters and grammar constraints.

#### LoRA Operations

```csharp
// Configure LoRA weights
public bool LoraWeight(List<LoraIdScale> loras)
public bool LoraWeight(params LoraIdScale[] loras)

// List available LoRA adapters
public List<LoraIdScalePath> LoraList()
```

**Description**: LoRA management with support for collections and parameter arrays.

#### Slot Management

```csharp
// Save/load slot state (inherited from LLMLocal)
public string SaveSlot(int idSlot, string filepath)
public string LoadSlot(int idSlot, string filepath)

// Cancel running request (inherited from LLMLocal)
public void Cancel(int idSlot)
```

**Description**: Slot operations for state persistence and request cancellation.

#### Utilities

```csharp
// Get embedding vector dimensions
public int EmbeddingSize()

// Static debugging methods
public static void Debug(int debugLevel)
public static void LoggingCallback(LlamaLib.CharArrayCallback callback)
public static void LoggingStop()
```

**Description**: Utility methods for embeddings information and global debugging configuration.

---

### LLMClient

#### Construction

```csharp
// Constructor for local LLM access
public LLMClient(LLMProvider provider)

// Constructor for remote LLM access
public LLMClient(string url, int port, string apiKey = "")
```

**Description**: Creates clients for local providers or remote HTTP endpoints with automatic resource management.

#### Core LLM Operations

```csharp
// All core LLM operations inherited from LLMLocal
// Including: Tokenize, Detokenize, Embeddings, Completion, GetTemplate, ApplyTemplate
```

**Description**: Full LLM functionality inherited from base classes, working transparently with local or remote backends.

---

### LLMAgent

#### Construction

```csharp
// Constructor for conversational agent
public LLMAgent(LLMLocal _llm, string _systemPrompt = "", string _userRole = "user", string _assistantRole = "assistant")
```

**Description**: Creates a conversational agent that wraps an LLMLocal instance with chat history management.

#### Core Chat Functionality

```csharp
// Conduct a chat interaction
public string Chat(string userPrompt, bool addToHistory = true, LlamaLib.CharArrayCallback callback = null, bool returnResponseJson = false)
public async Task<string> ChatAsync(string userPrompt, bool addToHistory = true, LlamaLib.CharArrayCallback callback = null, bool returnResponseJson = false)
```

**Description**: Main chat methods with sync/async support for conversational interactions.

#### Properties

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

**Description**: Properties for accessing and configuring agent state including slot assignment, roles, and conversation history.

#### History Management

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

**Description**: Comprehensive history management with strongly-typed ChatMessage objects and file persistence.

#### Slot-Aware Operations

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

**Description**: Slot-aware operations that automatically use the agent's assigned slot, hiding complexity from users.

</div>
</div>

<style>
.api-tabs {
  margin: 20px 0;
}

.tab-buttons {
  display: flex;
  border-bottom: 2px solid #e1e5e9;
  margin-bottom: 20px;
}

.tab-button {
  background: none;
  border: none;
  padding: 12px 24px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  color: #586069;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.tab-button:hover {
  color: #0366d6;
}

.tab-button.active {
  color: #0366d6;
  border-bottom-color: #0366d6;
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 20px 0;
}

th, td {
  border: 1px solid #e1e5e9;
  padding: 8px 12px;
  text-align: left;
}

th {
  background-color: #f6f8fa;
  font-weight: 600;
}

tr:nth-child(even) {
  background-color: #f9f9f9;
}

.method-category {
  font-weight: bold;
  background-color: #f1f3f4 !important;
}
</style>

<script>
function showTab(tabName) {
  // Hide all tab contents
  const contents = document.querySelectorAll('.tab-content');
  contents.forEach(content => {
    content.classList.remove('active');
  });
  
  // Remove active class from all buttons
  const buttons = document.querySelectorAll('.tab-button');
  buttons.forEach(button => {
    button.classList.remove('active');
  });
  
  // Show selected tab content
  document.getElementById(tabName + '-content').classList.add('active');
  
  // Add active class to clicked button
  event.target.classList.add('active');
}

// Initialize with C++ tab active
document.addEventListener('DOMContentLoaded', function() {
  showTab('cpp');
});
</script>