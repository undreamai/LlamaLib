# LLM Classes C++ API Documentation

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

## LLMService

Concrete implementation of LLMProvider with HTTP server functionality, parameter parsing, and integration with llama.cpp backend. This class provides a full-featured LLM service with HTTP server, parameter configuration, and backend integration with llama.cpp.

### Construction & Factory Methods

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

**Description**: Creates LLMService instances with various initialization methods. Factory methods allow creation from structured JSON parameters or command line arguments compatible with llama.cpp server. See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.

### Core LLM Operations

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

**Description**: Core language model operations for tokenization, text generation, embeddings, and template processing with optional streaming callbacks.

### Service Management

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

**Description**: Methods for managing the LLM service lifecycle and HTTP server functionality with optional API key authentication.

### Configuration

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

**Description**: Configuration methods for completion parameters, grammar constraints in GBNF format or JSON schema, chat templates, and SSL certificate setup for secure connections.

### LoRA Operations

```cpp
// Configure LoRA weights
bool lora_weight(const std::vector<LoraIdScale> &loras);

// List available LoRA adapters
std::vector<LoraIdScalePath> lora_list();
```

**Description**: Methods for managing Low-Rank Adaptation (LoRA) layers including weight configuration with scale factors and listing available adapter files with their paths.

### Slot Management

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

**Description**: Slot management for concurrent request processing, allowing state persistence to files and request cancellation for specific slots.

### Debugging & Utilities

```cpp
// Set debug level
void debug(int debug_level);

// Set logging callback function
void logging_callback(CharArrayFn callback);

// Get embedding vector dimensions
int embedding_size();
```

**Description**: Debugging and utility methods where debug_level controls verbosity (0 = off, 1 = LlamaLib messages, 2+ = llama.cpp messages), logging callbacks receive log messages, and embedding_size returns the number of dimensions in embedding vectors.

---

## LLMClient

Client for accessing LLM functionality locally or remotely with unified interface for both local LLMProvider instances and remote LLM services via HTTP. Provides a unified interface that can connect to either local LLMProvider instances or remote LLM services via HTTP, supporting all standard LLM operations including completion, tokenization, embeddings, and slot management.

### Construction

```cpp
// Constructor for local LLM access
LLMClient(LLMProvider *llm);

// Constructor for remote LLM access
LLMClient(const std::string &url, const int port, const std::string &API_key = "");
```

**Description**: Creates clients for either local LLMProvider instances or remote LLM services via HTTP with optional API key authentication.

### Core LLM Operations

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

**Description**: Unified interface for LLM operations that works transparently with both local and remote backends, providing the same API regardless of connection type.

### Slot Management

```cpp
// Get available processing slot (inherited from LLMLocal)
int get_next_available_slot();

// Cancel running request (inherited from LLMLocal)
void cancel(int id_slot);
```

**Description**: Slot management methods that work across local and remote connections for request processing and cancellation.

### Client-Specific Methods

```cpp
// Configure SSL certificate for remote connections
void set_SSL(const char *SSL_cert);

// Check if this is a remote client
bool is_remote() const;
```

**Description**: Client-specific configuration and utility methods where set_SSL configures certificate verification for remote HTTPS connections, and is_remote() returns true if configured for remote access.

---

## LLMAgent

High-level conversational agent that manages chat history and applies chat template formatting. Creates a conversation-aware interface that manages chat history and applies chat template formatting around an LLMLocal instance.

### Construction

```cpp
// Constructor for conversational agent
LLMAgent(LLMLocal *llm, const std::string &system_prompt = "", 
         const std::string &user_role = "user", const std::string &assistant_role = "assistant");
```

**Description**: Creates a conversational agent that manages conversations with the specified LLM backend, system prompt for context, and customizable role identifiers for users and assistants.

### Core Chat Functionality

```cpp
// Conduct a chat interaction
std::string chat(const std::string &user_prompt, bool add_to_history = true, 
                CharArrayFn callback = nullptr, bool return_response_json = false);
```

**Description**: Main chat method that processes user input, applies conversation context, generates a response, and optionally updates conversation history with streaming callback support.

### Slot Management

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

**Description**: Slot-aware methods that automatically use the agent's assigned slot for operations, including state persistence and request cancellation without requiring slot ID parameters.

### Role Configuration

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

**Description**: Configuration methods for chat roles and system prompts that define conversation context, where setting system prompt clears existing conversation history.

### History Management

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

**Description**: Comprehensive history management for persistent conversations including message addition with role-specific convenience methods, history manipulation, file-based persistence as JSON, and size tracking.