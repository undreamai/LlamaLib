#pragma once
#include "server.cpp"

class LLMParser {
    public:
        static bool readline(std::string & line);
        static std::vector<std::string> splitArguments(const std::string& inputString);
        static gpt_params string_to_gpt_params(std::string params_string);
};

class LLM {
    public:
        LLM(std::string params_string);
        LLM(int argc, char ** argv);
        ~LLM();
        
        server_context ctx_server;
        void run_server();

        std::string handle_tokenize(json body);
        std::string handle_detokenize(json body);
        std::string handle_completions(json data);

    private:
        gpt_params params;
        server_params sparams;
        std::thread server_thread;

        void parse_args(std::string params_string);
        void run_task();
};
