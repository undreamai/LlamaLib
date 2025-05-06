#include "undreamai.h"

int main(int argc, char ** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLM* llm = LLM_Construct(command.c_str());
    LLM_StartServer(llm);
    LLM_Start(llm);
    return 0;
}
