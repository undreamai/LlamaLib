#include "dynamic_loader.h"

int main(int argc, char** argv) {
    std::cout << GetPossibleArchitectures() << "\n";
    std::cout << GetPossibleArchitectures(true) << "\n";


    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }



    set_error_handlers();  // Set up crash signal handlers

    LLMBackend backend;
    LibHandle handle = nullptr;
    LLM* llm = nullptr;
    std::vector<std::string> possible_backends = {
        "libundreamai_avx2.dll",
        "undreamai_windows-hip.dll",
        "undreamai_windows-avx2.dll",
    };


    int result = tryLoadingBackend(possible_backends, command, backend, handle, llm);
    if (result == 0) {
        std::cout << "Successfully loaded and ran model." << std::endl;
    }
    else {
        std::cout << "Failed to load any backend." << std::endl;
    }

    backend.LLM_StartServer(llm);
    backend.LLM_Start(llm);
    return 0;
}

