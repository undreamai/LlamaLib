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
        LLMLib* lib;
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

    static LLMHandle from_LLMLib(LLMLib* ptr) {
        LLMHandle h;
        h.type = Type::Lib;
        h.lib = ptr;
        return h;
    }
};

#define CALL_LLM_FUNCTION(FUNC, HANDLE, ...)                            \
    do {                                                                \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::LLM:     FUNC((HANDLE).llm, ##__VA_ARGS__); break; \
        case LLMHandle::Type::LLMWithSlot:     FUNC((HANDLE).llmWithSlot, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Service: FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     LlamaLib_##FUNC((HANDLE).lib, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    } while (0)

#define CALL_LLMWITHSLOT_FUNCTION(FUNC, HANDLE, ...)                            \
    do {                                                                \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::LLMWithSlot:     FUNC((HANDLE).llmWithSlot, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Service: FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     LlamaLib_##FUNC((HANDLE).lib, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    } while (0)

#define CALL_LLM_PROVIDER_FUNCTION(FUNC, HANDLE, ...)                            \
    do {                                                                \
        switch ((HANDLE).type) {                                        \
        case LLMHandle::Type::Service: FUNC((HANDLE).service, ##__VA_ARGS__); break; \
        case LLMHandle::Type::Lib:     LlamaLib_##FUNC((HANDLE).lib, ##__VA_ARGS__); break; \
        default: throw std::runtime_error("Unknown LLMHandle type");    \
        }                                                               \
    } while (0)



void test_tokenization(LLMHandle handle, StringWrapper* wrapper) {
    std::cout << "LLM_Tokenize" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    CALL_LLM_FUNCTION(LLM_Tokenize, handle, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout << "LLM_Detokenize" << std::endl;
    CALL_LLM_FUNCTION(LLM_Detokenize, handle, reply.c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);
}

void test_completion(LLMHandle handle, StringWrapper* wrapper, bool stream) {
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

    CALL_LLM_FUNCTION(LLM_Completion, handle, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);

    std::string reply_data;
    if (stream)
        reply_data = concatenate_streaming_result(reply);
    else
        reply_data = json::parse(reply)["content"];
    ASSERT(reply_data != "");
}

void test_embedding(LLMHandle handle, StringWrapper* wrapper) {
    std::cout << "LLM_Embeddings" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    CALL_LLM_FUNCTION(LLM_Embeddings, handle, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);

    ASSERT(reply_data["embedding"].size() == EMBEDDING_SIZE);
}

void test_lora(LLMHandle handle, StringWrapper* wrapper) {
    std::cout << "LLM_Lora_List" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION(LLM_Lora_List, handle, wrapper);
    std::string reply = GetStringWrapperContent(wrapper);
    json reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);
}

void test_cancel(LLMHandle handle) {
    std::cout << "LLM_Cancel" << std::endl;
    CALL_LLMWITHSLOT_FUNCTION(LLM_Cancel, handle, ID_SLOT);
}

void test_slot_save_restore(LLMHandle handle, StringWrapper* wrapper) {
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

    CALL_LLMWITHSLOT_FUNCTION(LLM_Slot, handle, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    int n_saved = reply_data["n_saved"];
    ASSERT(n_saved > 0);

    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    std::cout << "LLM_Slot Restore" << std::endl;
    data["action"] = "restore";
    CALL_LLMWITHSLOT_FUNCTION(LLM_Slot, handle, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
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

LLMLib* start_llm_lib(std::string command)
{
    LLMLib* llmlib = Load_LLM_Library(command);
    if (!llmlib) {
        std::cerr << "Failed to load any backend." << std::endl;
        return nullptr;
    }
    llmlib->LLM_StartServer();
    llmlib->LLM_Start();
    return llmlib;
}

void stop_llm_service(LLMHandle handle)
{
    std::cout << "LLM_StopServer" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION(LLM_StopServer, handle);
    std::cout << "LLM_Stop" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION(LLM_Stop, handle);
    std::cout << "LLM_Delete" << std::endl;
    CALL_LLM_PROVIDER_FUNCTION(LLM_Delete, handle);
}

void run_tests(LLMHandle handle)
{
    StringWrapper* wrapper = StringWrapper_Construct();
    test_tokenization(handle, wrapper);
    test_completion(handle, wrapper, false);
    test_completion(handle, wrapper, true);
    test_embedding(handle, wrapper);
    if(handle.type == LLMHandle::Type::Lib || handle.type == LLMHandle::Type::Service)
    {
        test_lora(handle, wrapper);
    }
    if(handle.type != LLMHandle::Type::LLM)
    {
        test_cancel(handle);
        test_slot_save_restore(handle, wrapper);
    }
    delete wrapper;
}

int main(int argc, char** argv) {
    SetDebugLevel(ERR);

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
    LLM_StartServer(llm_service);
    RemoteLLMClient llm_remote_client("localhost", 8080);
    run_tests(LLMHandle::from_LLM(&llm_remote_client));

    stop_llm_service(LLMHandle::from_LLM_service(llm_service));

    std::cout << "-------- LLM lib --------" << std::endl;
    LLMLib* llmlib = start_llm_lib(command);
    run_tests(LLMHandle::from_LLMLib(llmlib));
    stop_llm_service(LLMHandle::from_LLMLib(llmlib));

    return 0;
}
