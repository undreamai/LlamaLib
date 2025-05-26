#include "LLM_lib.h"

int main(int argc, char** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLMLib* llmlib = Load_LLM_Library(command);
    if (!llmlib) {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }

    llmlib->LLM_StartServer();
    llmlib->LLM_Start();
    return 0;
}
