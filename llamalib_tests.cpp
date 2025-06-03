#include "LLM_service.h"
#include "LLM_client.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include <iostream>

std::string PROMPT = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
int ID_SLOT = 0;
int EMBEDDING_SIZE;

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << "\n" \
                      << "File: " << __FILE__ << "\n" \
                      << "Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)

// Trim from the start (left trim)
std::string ltrim(const std::string &s) {
    std::string result = s;
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return result;
}

// Trim from the end (right trim)
std::string rtrim(const std::string &s) {
    std::string result = s;
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());
    return result;
}

// Trim from both ends (left & right trim)
std::string trim(const std::string &s) {
    return ltrim(rtrim(s));
}

std::string concatenate_streaming_result(std::string input)
{
    std::vector<std::string> contents;
    std::istringstream stream(input);
    std::string line;

    std::string output = "";
    while (std::getline(stream, line)) {
        if (line.find("data: ") == 0) {
            std::string json_str = line.substr(6);
            try {
                json parsed = json::parse(json_str);
                output += parsed["content"];
            } catch (const json::exception& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
            }
        }
    }
    return output;
}

struct LLMHandle {
    enum class Type { LLM, LLMWithSlot, Service, Lib } type;
    union {
        LLM* llm;
        LLMWithSlot* llmWithSlot;
        LLMService* service;
#ifdef LLAMALIB_BUILD_RUNTIME_LIB
        LLMLib* lib;
#endif
    };

    static LLMHandle from_LLM(LLM* ptr) {
        LLMHandle h;
        h.type = Type::LLM;
        h.llm = ptr;
        return h;
    }

    static LLMHandle from_LLMWithSlot(LLMWithSlot* ptr) {
        LLMHandle h;
        h.type = Type::LLMWithSlot;
        h.llmWithSlot = ptr;
        return h;
    }

    static LLMHandle from_LLM_service(LLMService* ptr) {
        LLMHandle h;
        h.type = Type::Service;
        h.service = ptr;
        return h;
    }
#ifdef LLAMALIB_BUILD_RUNTIME_LIB 
    static LLMHandle from_LLMLib(LLMLib* ptr) {
        LLMHandle h;
        h.type = Type::Lib;
        h.lib = ptr;
        return h;
    }
#endif
};

#ifdef LLAMALIB_BUILD_RUNTIME_LIB
#pragma message("LLAMALIB_BUILD_RUNTIME_LIB is defined")
    #define CALL_LLMLIB_FUNCTION_NO_RET(FUNC, HANDLE, ...) LLMLib_##FUNC((HANDLE).lib, ##__VA_ARGS__); break;
    #define CALL_LLMLIB_FUNCTION(FUNC, HANDLE, ...) return LLMLib_##FUNC((HANDLE).lib, ##__VA_ARGS__); break;
#else
#pragma message("LLAMALIB_BUILD_RUNTIME_LIB is not defined")
    #define CALL_LLMLIB_FUNCTION(FUNC, HANDLE, ...) throw std::runtime_error("LLMLib not supported in this build");
    #define CALL_LLMLIB_FUNCTION_NO_RET(FUNC, HANDLE, ...) throw std::runtime_error("LLMLib not supported in this build");
#endif

#define CALL_LLM_FUNCTION(FUNC, HANDLE, ...)                            \
    ([&]() -> const char* {                                             \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::LLM:     return FUNC((HANDLE).llm, ##__VA_ARGS__); break; \
        case LLMHandle::Type::LLMWithSlot:     return FUNC((HANDLE).llmWithSlot, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Service: return FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     CALL_LLMLIB_FUNCTION(FUNC, HANDLE, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    })()

#define CALL_LLMWITHSLOT_FUNCTION(FUNC, HANDLE, ...)                    \
    ([&]() -> const char* {                                             \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::LLMWithSlot:     return FUNC((HANDLE).llmWithSlot, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Service: return FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     CALL_LLMLIB_FUNCTION(FUNC, HANDLE, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    })()

#define CALL_LLMWITHSLOT_FUNCTION_NO_RET(FUNC, HANDLE, ...)                             \
    do {                                                                 \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::LLMWithSlot:     FUNC((HANDLE).llmWithSlot, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Service: FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     CALL_LLMLIB_FUNCTION_NO_RET(FUNC, HANDLE, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                  \
    } while (0)

#define CALL_LLM_PROVIDER_FUNCTION(FUNC, HANDLE, ...)                   \
    ([&]() -> const char* {                                             \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::Service: return FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     CALL_LLMLIB_FUNCTION(FUNC, HANDLE, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    })()

#define CALL_LLM_PROVIDER_FUNCTION_NO_RET(FUNC, HANDLE, ...)                            \
    do {                                                                \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::Service: FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     CALL_LLMLIB_FUNCTION_NO_RET(FUNC, HANDLE, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    } while (0)


void test_tokenization(LLMHandle handle) {
    std::cout << "LLM_Tokenize" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    reply = std::string(CALL_LLM_FUNCTION(LLM_Tokenize, handle, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout << "LLM_Detokenize" << std::endl;
    reply = std::string(CALL_LLM_FUNCTION(LLM_Detokenize, handle, reply.c_str()));
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);
}

int counter = 0;

void count_calls(const char* c)
{
    counter++;
}

void test_completion(LLMHandle handle, bool stream) {
    std::cout << "LLM_Completion ( ";
    if (!stream) std::cout << "no ";
    std::cout << "streaming )" << std::endl;

    json data;
    std::string reply;

    data["id_slot"] = ID_SLOT;
    data["prompt"] = PROMPT;
    data["cache_prompt"] = true;
    data["n_predict"] = 10;
    data["n_keep"] = 30;
    data["stream"] = stream;

    counter = 0;
    reply = std::string(CALL_LLM_FUNCTION(LLM_Completion, handle, data.dump().c_str(), static_cast<CharArrayFn>(count_calls)));
    ASSERT(counter > int(stream));

    std::string reply_data;
    if (stream)
        reply_data = concatenate_streaming_result(reply);
    else
        reply_data = json::parse(reply)["content"];
    ASSERT(reply_data != "");
}

void test_embedding(LLMHandle handle) {
    std::cout << "LLM_Embeddings" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    reply = std::string(CALL_LLM_FUNCTION(LLM_Embeddings, handle, data.dump().c_str()));
    reply_data = json::parse(reply);

    ASSERT(reply_data["embedding"].size() == EMBEDDING_SIZE);
}

void test_lora(LLMHandle handle) {
    std::cout << "LLM_Lora_List" << std::endl;
    std::string reply;
    reply = std::string(CALL_LLM_PROVIDER_FUNCTION(LLM_Lora_List, handle));
    json reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);
}

void test_cancel(LLMHandle handle) {
    std::cout << "LLM_Cancel" << std::endl;
    CALL_LLMWITHSLOT_FUNCTION_NO_RET(LLM_Cancel, handle, ID_SLOT);
}

void test_slot_save_restore(LLMHandle handle) {
    std::cout << "LLM_Slot Save" << std::endl;
    std::string filename = "test_undreamai.save";
    json data;
    json reply_data;
    std::string reply;

    data["id_slot"] = ID_SLOT;
    data["action"] = "save";

#ifdef _WIN32
    char buffer[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, buffer);
    data["filepath"] = std::string(buffer) + "\\" + filename;
#else
    char buffer[PATH_MAX];
    getcwd(buffer, sizeof(buffer));
    data["filepath"] = std::string(buffer) + "/" + filename;
#endif

    reply = std::string(CALL_LLMWITHSLOT_FUNCTION(LLM_Slot, handle, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    int n_saved = reply_data["n_saved"];
    ASSERT(n_saved > 0);

    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    std::cout << "LLM_Slot Restore" << std::endl;
    data["action"] = "restore";
    reply = std::string(CALL_LLMWITHSLOT_FUNCTION(LLM_Slot, handle, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    ASSERT(reply_data["n_restored"] == n_saved);

    std::remove(filename.c_str());
}

LLMService* start_llm_service(const std::string& command)
{
    LLMService* llm_service = LLM_Construct(command.c_str());
    LLM_Start(llm_service);
    return llm_service;
}

#ifdef LLAMALIB_BUILD_RUNTIME_LIB
LLMLib* start_llm_lib(std::string command)
{
    LLMLib* llmlib = Load_LLM_Library(command);
    if (!llmlib) {
        std::cerr << "Failed to load any backend." << std::endl;
        return nullptr;
    }
    llmlib->LLM_Start_Server();
    llmlib->LLM_Start();
    return llmlib;
}
#endif

void stop_llm_service(LLMHandle handle)
{
    std::cout << "LLM_Stop_Server" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION_NO_RET(LLM_Stop_Server, handle);
    std::cout << "LLM_Stop" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION_NO_RET(LLM_Stop, handle);
    std::cout << "LLM_Delete" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION_NO_RET(LLM_Delete, handle);
}

class TestLLM : public LLMFull {
public:
    std::vector<int> TOKENS = std::vector<int>{1, 2, 3};
    std::string CONTENT = "my message";
    std::vector<float> EMBEDDING = std::vector<float>{0.1f, 0.2f, 0.3f};
    std::vector<LoraIdScalePath> LORAS = {
        {1, 1.0f, "model1.lora"},
        {2, 0.5f, "model2.lora"}
    };
    std::string SAVE_PATH = "test.save";

    std::string handle_tokenize_impl(const json& data) override {
        json result;
        result["tokens"] = TOKENS;
        return result.dump();
    }

    std::string handle_detokenize_impl(const json& data) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string handle_embeddings_impl(const json& data, httplib::Response*, std::function<bool()>) override {
        json result;
        result["embedding"] = EMBEDDING;
        return result.dump();
    }

    std::string handle_completions_impl(const json& data, CharArrayFn, httplib::Response*, std::function<bool()>, int) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string handle_slots_action_impl(const json& data, httplib::Response*) override {
        json result;
        result["filename"] = SAVE_PATH;
        return result.dump();
    }

    void handle_cancel_action_impl(int id_slot) override {
        cancel_called = true;
        cancelled_slot = id_slot;
    }

    std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response*) override {
        json result;
        result["success"] = true;
        return result.dump();
    }

    std::string handle_lora_adapters_list_impl() override {
        json j = json::array();
        for (auto& l : LORAS)
            j.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});
        return j.dump();
    }

    bool cancel_called = false;
    int cancelled_slot = -1;
};

void run_mock_tests() {
    TestLLM llm;

    // --- Tokenize ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"tokens", llm.TOKENS}};

        ASSERT(llm.build_tokenize_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_tokenize_json(output_json) == llm.TOKENS);
        ASSERT(json::parse(llm.handle_tokenize_json(llm.CONTENT))["tokens"] == llm.TOKENS);
        ASSERT(llm.handle_tokenize(llm.CONTENT.c_str()) == llm.TOKENS);
        ASSERT(llm.handle_tokenize(llm.CONTENT) == llm.TOKENS);
        ASSERT(llm.handle_tokenize(input_json) == llm.TOKENS);
        ASSERT(llm.handle_tokenize_json(llm.CONTENT.c_str()) == output_json.dump());
        ASSERT(llm.handle_tokenize_json(llm.CONTENT) == output_json.dump());
        ASSERT(llm.handle_tokenize_json(input_json) == output_json.dump());
    }

    // --- Detokenize ---
    {
        json input_json = {{"tokens", llm.TOKENS}};
        json output_json = {{"content", llm.CONTENT}};

        ASSERT(llm.build_detokenize_json(llm.TOKENS)["tokens"] == llm.TOKENS);
        ASSERT(llm.parse_detokenize_json(output_json) == llm.CONTENT);
        ASSERT(llm.handle_detokenize(llm.TOKENS) == llm.CONTENT);
        ASSERT(llm.handle_detokenize(input_json) == llm.CONTENT);
        ASSERT(llm.handle_detokenize_json(llm.TOKENS) == output_json.dump());
        ASSERT(llm.handle_detokenize_json(input_json) == output_json.dump());
    }

    // --- Embeddings ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"embedding", llm.EMBEDDING}};

        ASSERT(llm.build_embeddings_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_embeddings_json(output_json) == llm.EMBEDDING);
        ASSERT(json::parse(llm.handle_embeddings_json(llm.CONTENT))["embedding"] == llm.EMBEDDING);
        ASSERT(json::parse(llm.handle_embeddings_json(llm.CONTENT.c_str()))["embedding"] == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(llm.CONTENT) == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(llm.CONTENT.c_str()) == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(input_json) == llm.EMBEDDING);
    }

    // --- Completions ---
    {
        int id_slot = 1;
        json params = {{"temp", 0.7}};
        json input_json = {
            {"prompt", llm.CONTENT},
            {"id_slot", id_slot},
            {"temp", 0.7}
        };
        json output_json = {{"content", llm.CONTENT}};

        ASSERT(llm.build_completions_json(llm.CONTENT, id_slot, params) == input_json);
        ASSERT(llm.parse_completions_json(output_json) == llm.CONTENT);
        ASSERT(llm.handle_completions(llm.CONTENT, id_slot, params) == llm.CONTENT);
        ASSERT(llm.handle_completions(input_json) == llm.CONTENT);
        ASSERT(llm.handle_completions_json(llm.CONTENT, id_slot, params) == output_json.dump());
    }

    // --- Slots Action ---
    {
        int id_slot = 42;
        std::string action = "load";
        json input_json = {
            {"id_slot", id_slot},
            {"action", action},
            {"filepath", llm.SAVE_PATH}
        };
        json output_json = {{"filename", llm.SAVE_PATH}};

        ASSERT(llm.build_slots_action_json(id_slot, action, llm.SAVE_PATH) == input_json);
        ASSERT(llm.parse_slots_action_json(output_json) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action(id_slot, action, llm.SAVE_PATH) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action(input_json) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action_json(id_slot, action, llm.SAVE_PATH) == output_json.dump());
    }

    // --- LoRA Apply ---
    {
        std::vector<LoraIdScale> loras;
        for (auto& l : llm.LORAS)
            loras.push_back({l.id, l.scale});

        json input_json = json::array();
        for (const auto& l : loras)
            input_json.push_back({{"id", l.id}, {"scale", l.scale}});
        json output_json = {{"success", true}};

        ASSERT(llm.build_lora_adapters_apply_json(loras) == input_json);
        ASSERT(llm.parse_lora_adapters_apply_json(output_json));
        ASSERT(llm.handle_lora_adapters_apply_json(loras) == output_json.dump());
        ASSERT(llm.handle_lora_adapters_apply(input_json));
        ASSERT(llm.handle_lora_adapters_apply(loras));
    }

    // --- LoRA List ---
    {
        json input_json = json::array();
        for (const auto& l : llm.LORAS)
            input_json.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});

        ASSERT(llm.parse_lora_adapters_list_json(input_json) == llm.LORAS);
        ASSERT(llm.handle_lora_adapters_list() == llm.LORAS);
    }
}



void run_tests(LLMHandle handle)
{
    test_tokenization(handle);
    test_completion(handle, false);
    test_completion(handle, true);
    test_embedding(handle);
    if(handle.type == LLMHandle::Type::Lib || handle.type == LLMHandle::Type::Service)
    {
        test_lora(handle);
    }
    if(handle.type != LLMHandle::Type::LLM)
    {
        test_cancel(handle);
        test_slot_save_restore(handle);
    }
}


int main(int argc, char** argv) {
    SetDebugLevel(ERR);

    run_mock_tests();

    std::string command;
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLMService* llm_service = start_llm_service(command);
    EMBEDDING_SIZE = LLM_Embedding_Size(llm_service);

    std::cout << "-------- LLM service --------" << std::endl;
    run_tests(LLMHandle::from_LLMWithSlot(llm_service));

    std::cout << "-------- LLM client --------" << std::endl;
    LLMClient llm_client(llm_service);
    run_tests(LLMHandle::from_LLMWithSlot(&llm_client));

    std::cout << "-------- LLM remote client --------" << std::endl;
    LLM_Start_Server(llm_service);
    RemoteLLMClient llm_remote_client("localhost", 8080);
    run_tests(LLMHandle::from_LLM(&llm_remote_client));

    stop_llm_service(LLMHandle::from_LLM_service(llm_service));

#ifdef LLAMALIB_BUILD_RUNTIME_LIB
    std::cout << "-------- LLM lib --------" << std::endl;
    LLMLib* llmlib = start_llm_lib(command);
    run_tests(LLMHandle::from_LLMLib(llmlib));
    stop_llm_service(LLMHandle::from_LLMLib(llmlib));
#endif

    return 0;
}
