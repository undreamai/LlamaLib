#pragma once
#include "server.cpp"

class LLM {
    public:
        LLM(std::string params_string);
        LLM(int argc, char ** argv);
        ~LLM();

        static std::vector<std::string> splitArguments(const std::string& inputString);
        
        std::string handle_tokenize(json body);
        std::string handle_detokenize(json body);
        std::string handle_completions(json data);
        void handle_slots_action(json data);

    private:
        gpt_params params;
        server_params sparams;
        server_context ctx_server;
        std::thread server_thread;

        void parse_args(std::string params_string);
        void run_server();
};
