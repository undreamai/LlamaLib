#pragma once
#include "stringwrapper.h"
#include "server.cpp"

#include <setjmp.h>
#include <signal.h>
#ifdef _WIN32
    // Define custom equivalent for Windows (using SEH)
    #include <windows.h>
    #define sigjmp_buf jmp_buf
    #define sigsetjmp(jb, savemask) setjmp(jb)
    #define siglongjmp longjmp
#endif

int exit_code = 1;
int warning_code = -1;
int status;
std::string status_message;
sigjmp_buf point;

class LLM {
    public:
        LLM(std::string params_string);
        LLM(int argc, char ** argv);
        std::string chatTemplate;

        static std::vector<std::string> splitArguments(const std::string& inputString);
        int get_status();
        std::string get_status_message();
        
        std::string handle_template();
        std::string handle_tokenize(json body);
        std::string handle_detokenize(json body);
        std::string handle_completions(json data, StringWrapper* stringWrapper=nullptr, httplib::Response* res=nullptr);
        void handle_slots_action(json data);
        void handle_cancel_action(int id_slot);

        void start_server();
        void stop_server();
        void start_service();
        void stop_service();
        void set_template(const char* chatTemplate);
        bool is_running();

    private:
        gpt_params params;
        server_params sparams;
        server_context ctx_server;
        std::thread server_thread;
        std::unique_ptr<httplib::Server> svr;
        std::thread t;

        void parse_args(std::string params_string);
        void init(int argc, char ** argv);
        std::string handle_completions_non_streaming(int id_task, httplib::Response* res=nullptr);
        std::string handle_completions_streaming(int id_task, StringWrapper* stringWrapper=nullptr, httplib::DataSink* sink=nullptr);
};

#ifdef _WIN32
    #ifdef UNDREAMAI_EXPORTS
        #define UNDREAMAI_API __declspec(dllexport)
    #else
        #define UNDREAMAI_API __declspec(dllimport)
    #endif
#else
    #define UNDREAMAI_API
#endif

extern "C" {
    UNDREAMAI_API const void Logging(StringWrapper* wrapper);
    UNDREAMAI_API const void StopLogging();

	UNDREAMAI_API StringWrapper* StringWrapper_Construct();
	UNDREAMAI_API const void StringWrapper_Delete(StringWrapper* object);
	UNDREAMAI_API const int StringWrapper_GetStringSize(StringWrapper* object);
	UNDREAMAI_API const void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize, bool clear=false);

    UNDREAMAI_API LLM* LLM_Construct(const char* params_string);
    UNDREAMAI_API const void LLM_Delete(LLM* llm);
    UNDREAMAI_API const void LLM_Start(LLM* llm);
    UNDREAMAI_API const bool LLM_Started(LLM* llm);
    UNDREAMAI_API const void LLM_Stop(LLM* llm);
    UNDREAMAI_API const void LLM_StartServer(LLM* llm);
    UNDREAMAI_API const void LLM_StopServer(LLM* llm);
    UNDREAMAI_API const void LLM_SetTemplate(LLM* llm, const char* chatTemplate);
    UNDREAMAI_API const void LLM_Tokenize(LLM* llm, const char* json_data, StringWrapper* wrapper);
    UNDREAMAI_API const void LLM_Detokenize(LLM* llm, const char* json_data, StringWrapper* wrapper);
    UNDREAMAI_API const void LLM_Completion(LLM* llm, const char* json_data, StringWrapper* wrapper);
    UNDREAMAI_API const void LLM_Slot(LLM* llm, const char* json_data);
    UNDREAMAI_API const void LLM_Cancel(LLM* llm, int id_slot);
    UNDREAMAI_API const int LLM_Status(LLM* llm, StringWrapper* wrapper);
};