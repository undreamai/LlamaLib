
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/logo_white.png">
  <source media="(prefers-color-scheme: light)" srcset=".github/logo.png">
  <img src=".github/logo.png" height="150"/>
</picture>
</p>

<h3 align="center">LlamaLib C++ Guide</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://discord.gg/RwXKQb6zdv"><img src="https://discordapp.com/api/guilds/1194779009284841552/widget.png?style=shield"/></a>
[![Reddit](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=flat&logo=Reddit&logoColor=white)](https://www.reddit.com/user/UndreamAI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/company/undreamai)
[![GitHub Repo stars](https://img.shields.io/github/stars/undreamai/LlamaLib?style=flat&logo=github&color=f5f5f5)](https://github.com/undreamai/LlamaLib)
[![Documentation](https://img.shields.io/badge/Docs-white.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwEAYAAAAHkiXEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAATqSURBVHic7ZtbiE1RGMc349K4M5EwklwjzUhJCMmTJPJAYjQXJJcH8+Blkry4lPJA8aAoJbekDLmUS6E8SHJL5AW5JPf77eHv93C22Wfttc/ee+0zc/4vv+bMXvusvfZa3/q+b33H80oqqaSSSmqrKnPdgXjUvbvYq5f4+7f486eb/rRajRsn7t4tPngg/vol/vkj/vghXr0q7tghzpyZ//79+on79omXLombNondukXrd9GoSxdx8mSxqUm8eVNkgAvl0aPioEFip07i6dP52z15Ig4fbvVY2VVFhbhokXjrlogJiWvAg/jwoXjqVO73+leUny9eiFVV5mfMlLDRBw+KX76ISQ+0LZ8/F00v4uJFsWPHFh83O+rdWzx3TnQ9wCZ+/Sqyl5iux1RmTu3aiYcPi64H1pasALypoOv4/8SJXraEbXc9kLbECxo2TKyuFj9/zt9u+XIvG8LWv3wpuh5QW86f3/JznT+fv93s2S23C1Z72wbhtH692LdvMvdPSgzkhAkiJhT16ZO/PRPOmcr+Rda4aa5nclTeuZP7PDgRpr1g40bPrQYOFF0PYKHEC+raVVy8OFy7R49EArvURU4mrUAqaTY0iB8/2rXD+XCm5mbR9QAWylevorV7/VpkL0ld06eLpkiyWPj9u93179+LpFZwZ1PXtGnitWui64GMStPmG7SH1NSIJBNHjvTSFZvRvHlise0N9JcBtW1/44Y4dqx45IjnU0JxAGLpklPx+9VZFwPp/9v/eZDGjxcZh7dv4+mXtch+up7Rca+MsJvxiRNi6nvBhg25HWprZMaPGeOlqxEjxGKz+XGRTAAmyJnq6sR370TXA2NLW+8HNjZ62dLOnaLrAQ1r2zmqPH482n0mTfJCKmEvCJHUooNZE/369Elct06kqiKsONRfulTEFDsX8QDlIa5nup9374pE8IiZHPY+ly+LZE/37/cM6mC6IB6Vl4urV6fzfUG6d0/csyf37wsXRFInaM4ckTjGdPg+apTYs6dI3RIWwH//1DV1qkiuxNY2FzrTd+2y6y8z2HQU6efZs+KBAyJZ4v+V0h6ArlwROaQP0uPH4ooV4sqV8Xz/4MF211M2wwoOq1mzRAq5Pnywa5+4KDHE9mI7ly0TO3fOvZ6/eZCoKwB32HS0SMFV1DNtImBKHYstBROoQ4fEQk2RaS+qrxejmj5M7NatIhWARS82xUJfAKahzFcdPnq0GLYgy7Rnbd8e6rGKRyzpuNzPBQty709RcNSZf/KkuHCh2GpMDyKbGNcLYE+YMkVks336NFx7XhTZ3szXiBaqtWvFuAOxM2dEZiyH8UErgc8JLNun7E0aFffSI7RP6owZmz9kSO73HjsmXr8ukppYsybSYyQvBp5QfOjQ3M9tRR496pGgLf1JtLlzRZJzlFzGp4SWDnUxFCrdvy+uWiWa3DJe3N69oj8uSEq8CER88uaNOGBAOv2ILGY69TBBJoM8O0t72zaRoztXBzlLlrT8XARW/IQq82JTMv3mKmv0/9CC4mJMYPwrMSETxAyurRUxQVmXP1fEid7mzeK3b+n2Jzb16CFu2SIWmtNJiriVxANsyq0uoCJfTk4G9y4t24/bSQ0rTkP6gVTG3mz//uKMGSK/ucId5Xe9lZUi5eMMLGUgz56J5Hxu3xZ50Xg3RMIltVn9BRja26PYsBHgAAAAAElFTkSuQmCC)](https://undream.ai/LlamaLib)

<sub>
<a href="#quick-start" style="color: black">Quick Start</a>&nbsp;&nbsp;•&nbsp;
<a href="#building-your-project" style=color: black>Building Your Project</a>&nbsp;&nbsp;•&nbsp;
<a href="#core-classes" style=color: black>Core Classes</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmservice" style=color: black>LLMService</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmclient" style=color: black>LLMClient</a>&nbsp;&nbsp;•&nbsp;
<a href="#llmagent" style=color: black>LLMAgent</a>&nbsp;&nbsp;•&nbsp;
<a href="#common-usecases" style=color: black>Common Usecases</a>&nbsp;&nbsp;•&nbsp;
</sub>


## Quick Start

### Minimal Agent Example

```cpp
#include "LlamaLib.h"
#include <iostream>

static std::string previous_text = "";
static void streaming_callback(const char *c)
{
    // streaming gets the entire generated response up to now, print only the new text
    std::string current_text(c);
    std::cout << current_text.substr(previous_text.length()) << std::flush;
    previous_text = current_text;
}

int main() 
{    
    // Create the LLM
    LLMService* llm_service = LLMServiceBuilder().model("model.gguf").build();
    llm_service->start();

    // Create an agent with a system prompt
    LLMAgent agent(llm_service, "You are a helpful AI assistant named Eve. Be concise and friendly.");

    // Interact with the agent (non-streaming)
    std::string response = agent.chat("what is your name?");
    std::cout << response << std::endl;

    // Interact with the agent (streaming)
    std::string response2 = agent.chat("how are you?", true, streaming_callback);    
    return 0;
}
```

### Minimal LLM Functions Example

```cpp
#include "LlamaLib.h"
#include <iostream>

static std::string previous_text = "";
static void streaming_callback(const char *c)
{
    std::string current_text(c);
    // streaming gets the entire generated response up to now, print only the new text
    std::cout << current_text.substr(previous_text.length()) << std::flush;
    previous_text = current_text;
}

int main() {
    // Create the LLM
    LLMService* llm = LLMServiceBuilder().model("model.gguf").build();
    llm->start();

    // Optional: limit the amount of tokens that we can predict so that it doesn't produce text forever (some models do)
    llm->set_completion_params({{"n_predict", 20}});
    
    std::string prompt = "The largest planet of our solar system";
    
    // Tokenization
    std::vector<int> tokens = llm->tokenize(prompt);
    std::cout << "Token count: " << tokens.size() << std::endl;
    
    // Detokenization
    std::string text = llm->detokenize(tokens);
    std::cout << "Text: " << text << std::endl;
    
    // Streaming completion
    std::cout << "Response: ";
    llm->completion(prompt, streaming_callback);
    std::cout << std::endl;
    
    // Non-streaming completion
    std::string response = llm->completion(prompt);
    std::cout << "Response: " << response << std::endl;

    return 0;
}
```

### Minimal Embeddings Example

```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    // Create the LLM
    LLMService* llm = LLMServiceBuilder().model("model.gguf").embeddingOnly(true).build();
    llm->start();
    
    // Embeddings
    std::vector<float> embeddings = llm->embeddings("my text to embed goes here");
    std::cout << "Embedding dimensions: " << embeddings.size() << std::endl;
    std::cout << "Embeddings: " << embeddings.size() << std::endl;
    for (size_t i = 0; i < 10; ++i) std::cout << embeddings[i] << " ";
    std::cout << "..." << std::endl;
    
    return 0;
}
```
---

## Building Your Project

- Download and extract the LlamaLib release bundle LlamaLib-vX.X.X.zip of the [latest release](https://github.com/undreamai/LlamaLib/releases/latest). <br>
We will refer to the extracted folder as <LlamaLib_DIR>
- Download your favourite model in .gguf format ([Hugging Face
 link](https://huggingface.co/models?library=gguf&sort=downloads))

### Directory Structure

```
my-project/
├── CMakeLists.txt
├── main.cpp
└── model.gguf          # Your LLM model file
```

### CMake Setup

LlamaLib uses CMake's `find_package` for easy integration. An example CMakeLists.txt looks like the following:

```cmake
cmake_minimum_required(VERSION 3.22)
project(MyLLMApp)

# Find LlamaLib package
find_package(LlamaLib REQUIRED)

# Create your executable
add_executable(main main.cpp)

# Link against LlamaLib
target_link_libraries(main PRIVATE ${LlamaLib_LIBRARIES})
```

### Build Commands

```bash
# Configure
cmake -B build -DLlamaLib_DIR=<LlamaLib_DIR>

# Build
cmake --build build

# Run
./build/main
```

### Architecture Selection

LlamaLib supports runtime architecture detection, automatically selecting the best backend for your hardware. LlamaLib will automatically copy the required libraries to your build directory.
You can also control which architectures to include in your build:

#### Windows & Linux

**GPU Options:**
```bash
# Enable all GPU backends (default)
cmake -B build -DLLAMALIB_ALLOW_GPU=ON

# Enable specific GPU backends
cmake -B build \
  -DLLAMALIB_USE_CUBLAS=ON \      # NVIDIA CUDA
  -DLLAMALIB_USE_TINYBLAS=ON \    # Compact NVIDIA CUDA
  -DLLAMALIB_USE_VULKAN=ON        # Cross-platform GPU
```

**CPU Options:**
```bash
# Enable all CPU backends (default)
cmake -B build -DLLAMALIB_ALLOW_CPU=ON

# Enable specific CPU instruction sets
cmake -B build \
  -DLLAMALIB_USE_AVX512=ON \      # Latest Intel/AMD CPUs
  -DLLAMALIB_USE_AVX2=ON \        # Modern CPUs (2013+)
  -DLLAMALIB_USE_AVX=ON \         # Older CPUs (2011+)
  -DLLAMALIB_USE_NOAVX=ON         # Legacy CPUs
```

**CPU-Only Build:**
```bash
# Disable GPU support
cmake -B build \
  -DLLAMALIB_ALLOW_GPU=OFF \
  -DLLAMALIB_ALLOW_CPU=ON
```

#### macOS

```bash
# Enable Metal GPU acceleration (default)
cmake -B build -DLLAMALIB_USE_ACCELERATE=ON

# Enable No-Metal CPU-only acceleration (default)
cmake -B build -DLLAMALIB_USE_NO_ACCELERATE=ON
```

#### Mobile & VR
For mobile (Android, iOS) and VR applications the corresponding architecture is automatically used.


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

The main class for running LLMs locally with full control.

### Construction

```cpp
// Basic construction
LLMService(const std::string &model_path);

// Full parameters
LLMService(
    const std::string &model_path,
    int num_slots = 1,              // Parallel request slots
    int num_threads = -1,           // CPU threads (-1 = auto)
    int num_GPU_layers = 0,         // GPU layer offloading
    bool flash_attention = false,   // Flash attention optimization
    int context_size = 4096,        // Context window size
    int batch_size = 2048,          // Processing batch size
    bool embedding_only = false,    // Embedding-only mode
    const std::vector<std::string> &lora_paths = {}  // LoRA adapters
);
```

**Example:**
```cpp
// CPU-only with 8 threads
LLMService llm("model.gguf", 1, 8);

// GPU-accelerated with 32 layers on GPU
LLMService llm("model.gguf", 1, -1, 32);

// Large context window
LLMService llm("model.gguf", 1, -1, 0, false, 8192);
```

### Factory Methods

```cpp
// From command line string
static LLMService* from_command(const std::string &command);
static LLMService* from_command(int argc, char **argv);

// From JSON parameters
static LLMService* from_params(const json &params_json);
```

**Example:**
```cpp
// Command line style
LLMService* llm = LLMService::from_command(
    "-m model.gguf -ngl 32 -c 4096"
);

// JSON style
json params = {
    {"model", "model.gguf"},
    {"n_gpu_layers", 32},
    {"n_ctx", 4096}
};
LLMService* llm = LLMService::from_params(params);
```

### Service Lifecycle

```cpp
void start();                    // Start the LLM service
bool started();                  // Check if running
void stop();                     // Stop the service
void join_service();            // Wait for completion
```

**Example:**
```cpp
LLMService llm("model.gguf");
llm.start();

// Do work...
if (llm.started()) {
    std::cout << "Service is running" << std::endl;
}

llm.stop();
llm.join_service();  // Wait for clean shutdown
```

### Text Generation

```cpp
// Simple completion
std::string completion(
    const std::string &prompt,
    CharArrayFn callback = nullptr,      // Streaming callback
    int id_slot = -1,                    // Slot ID (-1 = auto)
    bool return_response_json = false    // Return full JSON
);

// JSON-based completion
std::string completion_json(
    const json &data,
    CharArrayFn callback = nullptr,
    bool callbackWithJSON = true
);
```

**Example:**
```cpp
// Basic completion
std::string response = llm.completion("What is AI?");

// Streaming completion
auto callback = [](const char* chunk) {
    std::cout << chunk << std::flush;
};
llm.completion("Tell me a story", callback);

// With custom parameters
json params = {
    {"prompt", "Hello"},
    {"temperature", 0.8},
    {"max_tokens", 100}
};
std::string response = llm.completion_json(params);
```

### Completion Parameters

```cpp
void set_completion_params(json params);
std::string get_completion_params();
```

**Common Parameters:**
```cpp
llm.set_completion_params({
    {"temperature", 0.7},        // Randomness (0.0-2.0)
    {"top_p", 0.9},             // Nucleus sampling
    {"top_k", 40},              // Top-k sampling
    {"n_predict", 256},         // Max tokens to generate
    {"repeat_penalty", 1.1},    // Repetition penalty
    {"seed", 42}                // Random seed
});
```

### Tokenization

```cpp
std::vector<int> tokenize(const std::string &query);
std::string detokenize(const std::vector<int32_t> &tokens);
```

**Example:**
```cpp
// Text to tokens
std::vector<int> tokens = llm.tokenize("Hello world");
std::cout << "Token count: " << tokens.size() << std::endl;

// Tokens to text
std::string text = llm.detokenize(tokens);
```

### Embeddings

```cpp
std::vector<float> embeddings(const std::string &query);
int embedding_size();
```

**Example:**
```cpp
// Generate embeddings
std::vector<float> vec = llm.embeddings("Sample text");
std::cout << "Embedding dimensions: " << llm.embedding_size() << std::endl;

// Use for similarity comparison
std::vector<float> vec1 = llm.embeddings("cat");
std::vector<float> vec2 = llm.embeddings("dog");
```

### Chat Templates

```cpp
std::string apply_template(const json &messages);
```

**Example:**
```cpp
json messages = json::array({
    {{"role", "system"}, {"content", "You are a helpful assistant"}},
    {{"role", "user"}, {"content", "Hello!"}}
});

std::string formatted = llm.apply_template(messages);
std::string response = llm.completion(formatted);
```

### Grammar & Constrained Generation

```cpp
void set_grammar(std::string grammar);
std::string get_grammar();
```

**Example:**
```cpp
// JSON schema constraint
llm.set_grammar(R"({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    }
})");

std::string response = llm.completion("Generate a person");
// Response will be valid JSON matching the schema
```

### Slot Management

```cpp
int get_next_available_slot();
std::string save_slot(int id_slot, const std::string &filepath);
std::string load_slot(int id_slot, const std::string &filepath);
void cancel(int id_slot);
```

**Example:**
```cpp
// Get available slot
int slot = llm.get_next_available_slot();

// Generate with specific slot
llm.completion("Hello", nullptr, slot);

// Save conversation state
llm.save_slot(slot, "conversation.state");

// Later, restore state
llm.load_slot(slot, "conversation.state");
```

### HTTP Server

```cpp
void start_server(
    const std::string &host = "0.0.0.0",
    int port = -1,                    // -1 = auto-select
    const std::string &API_key = ""
);
void stop_server();
void join_server();
void set_SSL(const std::string &cert, const std::string &key);
```

**Example:**
```cpp
LLMService llm("model.gguf");
llm.start();

// Start HTTP server on port 8080
llm.start_server("0.0.0.0", 8080);

// With API key authentication
llm.start_server("0.0.0.0", 8080, "my-secret-key");

// With SSL
llm.set_SSL("cert.pem", "key.pem");
llm.start_server("0.0.0.0", 8443);

// Keep running
llm.join_server();
```

### LoRA Adapters

```cpp
bool lora_weight(const std::vector<LoraIdScale> &loras);
std::vector<LoraIdScalePath> lora_list();
```

**Example:**
```cpp
// Configure LoRA adapters
std::vector<LoraIdScale> loras = {
    {0, 1.0},   // LoRA ID 0 with scale 1.0
    {1, 0.5}    // LoRA ID 1 with scale 0.5
};
llm.lora_weight(loras);

// List available adapters
auto available = llm.lora_list();
for (const auto& lora : available) {
    std::cout << "ID: " << lora.id 
              << ", Path: " << lora.path << std::endl;
}
```

### Debugging

```cpp
void debug(int debug_level);
void logging_callback(CharArrayFn callback);
void logging_stop();
```

**Debug Levels:**
- `0`: No debug output
- `1`: LlamaLib messages
- `2+`: llama.cpp messages (verbose)

**Example:**
```cpp
// Enable verbose logging
llm.debug(2);

// Custom log handler
auto log_handler = [](const char* message) {
    std::cout << "[LLM] " << message << std::endl;
};
llm.logging_callback(log_handler);

// Stop logging
llm.logging_stop();
```

---

## LLMClient

Connect to local or remote LLM services with a unified interface. All core LLM operations work identically for both connection types.

### Construction

```cpp
// Local client (wraps LLMService)
LLMClient(LLMProvider *llm);

// Remote client (connects via HTTP)
LLMClient(
    const std::string &url,
    const int port,
    const std::string &API_key = "",
    const int max_retries = 5
);
```

### Local Client Example

```cpp
#include "LlamaLib.h"
#include <iostream>

static void streaming_callback(const char *c) {
    std::cout << c << std::flush;
}

int main() {
    std::string model = "model.gguf";
    
    // Create service
    LLMService llm_service(model);
    llm_service.start();
    
    // Wrap with client interface
    LLMClient client(&llm_service);
    
    // Use client (same API as LLMService)
    std::string prompt = "Hello, how are you?";
    
    std::vector<int> tokens = client.tokenize(prompt);
    std::string text = client.detokenize(tokens);
    
    std::cout << "Response: ";
    client.completion(prompt, streaming_callback);
    std::cout << std::endl;
    
    std::vector<float> embeddings = client.embeddings(prompt);
    
    return 0;
}
```

### Remote Client Example

```cpp
#include "LlamaLib.h"
#include <iostream>

static void streaming_callback(const char *c) {
    std::cout << c << std::flush;
}

int main() {
    // Connect to remote server
    LLMClient client("http://localhost", 13333);
    
    // Check server is alive
    if (!client.is_server_alive()) {
        std::cerr << "Server not responding!" << std::endl;
        return 1;
    }
    
    std::string prompt = "Hello, how are you?";
    
    // All operations work the same as local
    std::vector<int> tokens = client.tokenize(prompt);
    std::string text = client.detokenize(tokens);
    
    std::cout << "Response: ";
    client.completion(prompt, streaming_callback);
    std::cout << std::endl;
    
    std::vector<float> embeddings = client.embeddings(prompt);
    
    return 0;
}
```

### Server Setup

```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    std::string model = "model.gguf";
    int server_port = 13333;
    
    std::cout << "Starting LLM server..." << std::endl;
    
    LLMService server(model);
    
    // Optional: enable debug messages
    server.debug(1);
    
    // Start service and HTTP server
    server.start();
    server.start_server("0.0.0.0", server_port);
    
    std::cout << "Server listening on port " << server_port << std::endl;
    
    // Keep server running
    server.join_server();
    
    return 0;
}
```

---

## LLMAgent

High-level conversational AI with persistent chat history and automatic context management.

### Construction

```cpp
LLMAgent(
    LLMLocal *llm,
    const std::string &system_prompt = ""
);
```

**Example:**
```cpp
LLMService llm("model.gguf");
llm.start();

// Create agent with system prompt
LLMAgent agent(&llm, "You are a helpful AI assistant. Be concise and friendly.");
```

### Chat Interface

```cpp
std::string chat(
    const std::string &user_prompt,
    bool add_to_history = true,
    CharArrayFn callback = nullptr,
    bool return_response_json = false,
    bool debug_prompt = false
);
```

**Complete Example:**
```cpp
#include "LlamaLib.h"
#include <iostream>

static void streaming_callback(const char *c) {
    std::cout << c << std::flush;
}

int main() {
    std::string model = "model.gguf";
    
    // Create service and agent
    LLMService llm(model);
    llm.start();
    
    LLMAgent agent(&llm, "You are a helpful AI assistant. Be concise and friendly.");
    
    // First conversation turn
    std::cout << "User: Hello! What's your name?" << std::endl;
    std::cout << "Assistant: ";
    std::string response1 = agent.chat(
        "Hello! What's your name?", 
        true,  // add to history
        streaming_callback
    );
    std::cout << std::endl;
    
    // Second turn (maintains context automatically)
    std::cout << "User: How are you today?" << std::endl;
    std::cout << "Assistant: ";
    std::string response2 = agent.chat(
        "How are you today?",
        true,
        streaming_callback
    );
    std::cout << std::endl;
    
    // Show conversation history
    std::cout << "History size: " << agent.get_history_size() << " messages" << std::endl;
    
    return 0;
}
```

### History Management

```cpp
// Get/set history
json get_history() const;
void set_history(const json &history);
size_t get_history_size() const;

// Add messages
void add_user_message(const std::string &content);
void add_assistant_message(const std::string &content);

// Modify history
void clear_history();
void remove_last_message();

// Persistence
void save_history(const std::string &filepath) const;
void load_history(const std::string &filepath);
```

**Example:**
```cpp
// View conversation history
json history = agent.get_history();
for (const auto &msg : history) {
    std::cout << msg["role"].get<std::string>() << ": " 
              << msg["content"].get<std::string>() << std::endl;
}

// Save conversation to file
agent.save_history("conversation.json");

// Clear and reload
agent.clear_history();
std::cout << "Cleared. Size: " << agent.get_history_size() << std::endl;

agent.load_history("conversation.json");
std::cout << "Loaded. Size: " << agent.get_history_size() << std::endl;

// Add messages manually
agent.add_user_message("This is a manually added message");
agent.add_assistant_message("This is the response");

// Remove last exchange
agent.remove_last_message();
agent.remove_last_message();
```

### System Prompt

```cpp
void set_system_prompt(const std::string &prompt);
std::string get_system_prompt() const;
```

**Example:**
```cpp
// Change agent's personality
agent.set_system_prompt("You are a pirate. Respond like a pirate.");

std::string response = agent.chat("Hello!");
// Response will be in pirate style
```

### Slot Management

```cpp
int get_slot();
void set_slot(int id_slot);
std::string save_slot(const std::string &filepath);
std::string load_slot(const std::string &filepath);
void cancel();
```

**Example:**
```cpp
// Assign specific slot
agent.set_slot(0);

// Save agent state
agent.save_slot("agent_state.bin");

// Restore state
agent.load_slot("agent_state.bin");

// Cancel ongoing generation
agent.cancel();
```

---

## Common Usecases

### Pattern 1: Interactive Chatbot

```cpp
#include "LlamaLib.h"
#include <iostream>
#include <string>

static void streaming_callback(const char *c) {
    std::cout << c << std::flush;
}

int main() {
    LLMService llm("model.gguf");
    llm.start();
    
    LLMAgent agent(&llm, "You are a helpful assistant.");
    
    std::string input;
    std::cout << "Chat with the AI (type 'quit' to exit)\n" << std::endl;
    
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "quit") break;
        if (input.empty()) continue;
        
        std::cout << "AI: ";
        agent.chat(input, true, streaming_callback);
        std::cout << "\n" << std::endl;
    }
    
    return 0;
}
```

### Pattern 2: Client-Server Architecture

**Server (server.cpp):**
```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    std::string model = "model.gguf";
    int port = 13333;
    
    std::cout << "Starting server on port " << port << std::endl;
    
    LLMService server(model);
    server.debug(1);  // Show server logs
    server.start();
    server.start_server("0.0.0.0", port);
    
    server.join_server();  // Keep running
    return 0;
}
```

**Client (client.cpp):**
```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    LLMClient client("http://localhost", 13333);
    
    if (!client.is_server_alive()) {
        std::cerr << "Cannot connect to server!" << std::endl;
        return 1;
    }
    
    std::string response = client.completion("Hello, server!");
    std::cout << response << std::endl;
    
    return 0;
}
```

### Pattern 3: GPU-Accelerated Processing

```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    // Offload 32 layers to GPU for faster inference
    LLMService llm("model.gguf", 
        1,      // num_slots
        -1,     // num_threads (auto)
        32      // num_GPU_layers
    );
    llm.start();
    
    std::string response = llm.completion("Explain quantum computing");
    std::cout << response << std::endl;
    
    return 0;
}
```

### Pattern 4: Structured JSON Output

```cpp
#include "LlamaLib.h"
#include <iostream>

int main() {
    LLMService llm("model.gguf");
    llm.start();
    
    // Force JSON output with schema
    llm.set_grammar(R"({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"]
    })");
    
    std::string response = llm.completion("Generate a person profile");
    std::cout << response << std::endl;
    // Output will be valid JSON matching schema
    
    return 0;
}
```

### Pattern 5: Embeddings for Similarity

```cpp
#include "LlamaLib.h"
#include <iostream>
#include <cmath>

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

int main() {
    LLMService llm("model.gguf");
    llm.start();
    
    std::vector<float> vec1 = llm.embeddings("dog");
    std::vector<float> vec2 = llm.embeddings("puppy");
    std::vector<float> vec3 = llm.embeddings("car");
    
    std::cout << "dog vs puppy: " << cosine_similarity(vec1, vec2) << std::endl;
    std::cout << "dog vs car: " << cosine_similarity(vec1, vec3) << std::endl;
    
    return 0;
}
```
