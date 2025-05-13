#include "dynamic_loader.h"

int main(int argc, char** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLMLib* llmlib = Load_LLM_Library(TINYBLAS, command);
    if (llmlib) {
        std::cout << "Successfully loaded and ran model." << std::endl;
    }
    else {
        std::cout << "Failed to load any backend." << std::endl;
    }

    llmlib->LLM_StartServer();
    llmlib->LLM_Start();
    return 0;
}

