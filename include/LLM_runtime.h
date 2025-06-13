#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <setjmp.h>

#include "defs.h"
#include "error_handling.h"
#include "LLM.h"

#if defined(_WIN32) || defined(__linux__)
#include "archchecker.h"
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
#include <windows.h>
using LibHandle = HMODULE;
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_SYM(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIB(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
using LibHandle = void*;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY)
#define GET_SYM(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#endif

//=================================== FUNCTION LISTS ===================================//

class LLMService;

#define LLM_FUNCTIONS_LIST(M) \
    M(LLM_Construct,       LLMService*, const char*) \
    M(LLM_Tokenize,        const char*, LLM*, const char*) \
    M(LLM_Detokenize,      const char*, LLM*, const char*) \
    M(LLM_Embeddings,      const char*, LLM*, const char*) \
    M(LLM_Completion,      const char*, LLM*, const char*, CharArrayFn) \
    M(LLM_Slot,            const char*, LLMWithSlot*, const char*) \
    M(LLM_Cancel,          void,        LLMWithSlot*, int) \
    M(LLM_Lora_Weight,     const char*, LLMProvider*, const char*) \
    M(LLM_Lora_List,       const char*, LLMProvider*) \
    M(LLM_Delete,          void,        LLMProvider*) \
    M(LLM_Start,           void,        LLMProvider*) \
    M(LLM_Started,         bool,        LLMProvider*) \
    M(LLM_Stop,            void,        LLMProvider*) \
    M(LLM_Start_Server,    void,        LLMProvider*) \
    M(LLM_Stop_Server,     void,        LLMProvider*) \
    M(LLM_Join_Service,    void,        LLMProvider*) \
    M(LLM_Join_Server,     void,        LLMProvider*) \
    M(LLM_SetSSL,          void,        LLMProvider*, const char*, const char*) \
    M(LLM_Status_Code,     int,         LLMProvider*) \
    M(LLM_Status_Message,  const char*, LLMProvider*) \
    M(LLM_Embedding_Size,  int,         LLMProvider*)

class UNDREAMAI_API LLMRuntime : public LLMProvider {
public:
    LLMRuntime(const std::string& command, const std::string& path = "");
    LLMRuntime(const char* command, const std::string& path = "");
    LLMRuntime(int argc, char ** argv, const std::string& path = "");
    ~LLMRuntime();

    LibHandle handle = nullptr;
    LLMProvider* llm = nullptr;

    bool create_LLM_library_from_path(const std::string& command, const std::string& path);
    bool create_LLM_library(const std::string& command, const std::string& path="");

    //=================================== LLM METHODS START ===================================//

    std::string handle_tokenize_impl(const json& data) override {
        return LLM_Tokenize((LLM*)llm, data.dump().c_str());
    }

    std::string handle_detokenize_impl(const json& data) override {
        return LLM_Detokenize((LLM*)llm, data.dump().c_str());
    }

    std::string handle_embeddings_impl(const json& data, httplib::Response*, std::function<bool()> = [] { return false; }) override {
        return LLM_Embeddings((LLM*)llm, data.dump().c_str());
    }
    std::string handle_completions_impl(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override {
        return LLM_Completion((LLM*)llm, data.dump().c_str(), callback);
    }

    std::string handle_slots_action_impl(const json& data, httplib::Response* = nullptr) override {
        return LLM_Slot((LLMWithSlot*)llm, data.dump().c_str());
    }

    void handle_cancel_action_impl(int id_slot) override {
        LLM_Cancel((LLMWithSlot*)llm, id_slot);
    }

    std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response* = nullptr) override {
        return LLM_Lora_Weight((LLMProvider*)llm, data.dump().c_str());
    }

    std::string handle_lora_adapters_list_impl() override {
        return LLM_Lora_List((LLMProvider*)llm);
    }

    int get_status() override {
        return LLM_Status_Code((LLMProvider*)llm);
    }

    std::string get_status_message() override {
        return LLM_Status_Message((LLMProvider*)llm);
    }

    void start_server() override { LLM_Start_Server((LLMProvider*)llm); }
    void stop_server() override { LLM_Stop_Server((LLMProvider*)llm); }
    void join_server() override { LLM_Join_Server((LLMProvider*)llm); }
    void start_service() override { LLM_Start((LLMProvider*)llm); }
    void stop_service() override { LLM_Stop((LLMProvider*)llm); }
    void join_service() override { LLM_Join_Service((LLMProvider*)llm); }
    void set_SSL(const char* cert, const char* key) override { LLM_SetSSL((LLMProvider*)llm, cert, key); }
    bool is_running() override { return LLM_Started((LLMProvider*)llm); }

    int embedding_size() override {
        return LLM_Embedding_Size((LLMProvider*)llm);
    }

    //=================================== LLM METHODS END ===================================//

#define DECLARE_FN(name, ret, ...) \
    ret (*name)(__VA_ARGS__) = nullptr;
    LLM_FUNCTIONS_LIST(DECLARE_FN)
#undef DECLARE_FN
};

const std::vector<std::string> available_architectures(bool gpu);

//=================================== EXTERNAL API ===================================//

extern "C" {
    UNDREAMAI_API const char* Available_Architectures(bool gpu);
    UNDREAMAI_API LLMRuntime* LLMRuntime_Construct(const std::string& command, const std::string& path="");
}
