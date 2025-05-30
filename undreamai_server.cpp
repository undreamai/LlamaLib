#include "LLM_service.h"

int main(int argc, char ** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLMService* llm = LLM_Construct(command.c_str());
    LLM_StartServer(llm);
    LLM_Start(llm);
    llm->join_server();
    return 0;
}
