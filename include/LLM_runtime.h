#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <setjmp.h>
#include <type_traits>
#include <algorithm>
#include <cstdlib>

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
#include <libloaderapi.h>
using LibHandle = HMODULE;
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_SYM(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIB(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
#include <unistd.h>
#include <limits.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using LibHandle = void*;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY)
#define GET_SYM(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#endif

//=================================== FUNCTION LISTS ===================================//

class LLMService;
#define LLM_FUNCTIONS_LIST(M) \
    M(LLMService_Construct,     LLMService*, const char*, int, int, int, bool, int, int, bool, int, const char**) \
    M(LLMService_From_Command,  LLMService*, const char*) \
    M(LLM_Tokenize,        const char*, LLM*, const char*) \
    M(LLM_Detokenize,      const char*, LLM*, const char*) \
    M(LLM_Embeddings,      const char*, LLM*, const char*) \
    M(LLM_Completion,      const char*, LLM*, const char*, CharArrayFn) \
    M(LLM_Slot,            const char*, LLMLocal*, const char*) \
    M(LLM_Cancel,          void,        LLMLocal*, int) \
    M(LLM_Lora_Weight,     const char*, LLMProvider*, const char*) \
    M(LLM_Lora_List,       const char*, LLMProvider*) \
    M(LLM_Delete,          void,        LLMProvider*) \
    M(LLM_Start,           void,        LLMProvider*) \
    M(LLM_Started,         bool,        LLMProvider*) \
    M(LLM_Stop,            void,        LLMProvider*) \
    M(LLM_Start_Server,    void,        LLMProvider*, const char*, int, const char*) \
    M(LLM_Stop_Server,     void,        LLMProvider*) \
    M(LLM_Join_Service,    void,        LLMProvider*) \
    M(LLM_Join_Server,     void,        LLMProvider*) \
    M(LLM_Set_SSL,         void,        LLMProvider*, const char*, const char*) \
    M(LLM_Status_Code,     int,         LLMProvider*) \
    M(LLM_Status_Message,  const char*, LLMProvider*) \
    M(LLM_Embedding_Size,  int,         LLMProvider*)

class UNDREAMAI_API LLMRuntime : public LLMProvider {
public:
    LLMRuntime();
    LLMRuntime(const std::string& model_path, int num_threads=-1, int num_GPU_layers=0, int num_parallel=1, bool flash_attention=false, int context_size=4096, int batch_size=2048, bool embedding_only=false, const std::vector<std::string>& lora_paths = {});
    ~LLMRuntime();

    static LLMRuntime* from_command(const std::string& command);
    static LLMRuntime* from_command(int argc, char ** argv);
    
    LibHandle handle = nullptr;
    LLMProvider* llm = nullptr;

    bool create_LLM_library(const std::string& command);

    //=================================== LLM METHODS START ===================================//
    void start_server(const std::string& host="0.0.0.0", int port=0, const std::string& API_key="") override { ((LLMProvider*)llm)->start_server(host, port, API_key); }
    void stop_server() override { ((LLMProvider*)llm)->stop_server(); }
    void join_server() override { ((LLMProvider*)llm)->join_server(); }
    void start() override { ((LLMProvider*)llm)->start(); }
    void stop() override { ((LLMProvider*)llm)->stop();; }
    void join_service() override { ((LLMProvider*)llm)->join_service(); }
    void set_SSL(const std::string& cert, const std::string& key) override { ((LLMProvider*)llm)->set_SSL(cert, key); }
    bool started() override { return ((LLMProvider*)llm)->started(); }
    int embedding_size() override { return ((LLMProvider*)llm)->embedding_size();}
    //=================================== LLM METHODS END ===================================//

#define DECLARE_FN(name, ret, ...) \
    ret (*name)(__VA_ARGS__) = nullptr;
    LLM_FUNCTIONS_LIST(DECLARE_FN)
#undef DECLARE_FN

protected:
    std::vector<std::filesystem::path> search_paths;

    //=================================== LLM METHODS START ===================================//
    std::string tokenize_json(const json& data) override {
        return LLM_Tokenize((LLM*)llm, data.dump().c_str());
    }

    std::string detokenize_json(const json& data) override {
        return LLM_Detokenize((LLM*)llm, data.dump().c_str());
    }

    std::string embeddings_json(const json& data) override {
        return LLM_Embeddings((LLM*)llm, data.dump().c_str());
    }
    std::string completion_json(const json& data, CharArrayFn callback = nullptr) override {
        return LLM_Completion((LLM*)llm, data.dump().c_str(), callback);
    }

    std::string slot_json(const json& data) override {
        return LLM_Slot((LLMLocal*)llm, data.dump().c_str());
    }

    void cancel(int id_slot) override {
        LLM_Cancel((LLMLocal*)llm, id_slot);
    }

    std::string lora_weight_json(const json& data) override {
        return LLM_Lora_Weight((LLMProvider*)llm, data.dump().c_str());
    }

    std::string lora_list_json() override {
        return LLM_Lora_List((LLMProvider*)llm);
    }
    //=================================== LLM METHODS END ===================================//

    bool create_LLM_library_backend(const std::string& command, const std::string& llm_lib_filename);
};

const std::string os_library_dir();
const std::vector<std::string> available_architectures(bool gpu);
static std::filesystem::path get_executable_directory();
static std::filesystem::path get_current_directory();
static std::vector<std::filesystem::path> get_env_library_paths(const std::vector<std::string>& env_vars);
static std::vector<std::filesystem::path> get_search_directories();
std::vector<std::string> get_default_library_env_vars();

//=================================== EXTERNAL API ===================================//

extern "C" {
    UNDREAMAI_API const char* Available_Architectures(bool gpu);
    UNDREAMAI_API LLMRuntime* LLMRuntime_Construct(const char* model_path, int num_threads=-1, int num_GPU_layers=0, int num_parallel=1, bool flash_attention=false, int context_size=4096, int batch_size=2048, bool embedding_only=false, int lora_count=0, const char** lora_paths=nullptr);
    UNDREAMAI_API LLMRuntime* LLMRuntime_From_Command(const char* command);
}
