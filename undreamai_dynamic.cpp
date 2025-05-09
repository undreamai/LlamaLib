#include "dynamic_loader.h"

int main(int argc, char** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLMBackend backend;
    LibHandle handle = nullptr;
    LLM* llm = nullptr;

    int result = load_backends_fallback(NO_GPU, command, backend);
    if (result == 0) {
        std::cout << "Successfully loaded and ran model." << std::endl;
    }
    else {
        std::cout << "Failed to load any backend." << std::endl;
    }

    backend.LLM_StartServer();
    backend.LLM_Start();
    return 0;
}

