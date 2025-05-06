#include "dynamic_loader.h"

int main(int argc, char** argv) {
    std::string libPath = "undreamai_windows-avx2.dll"; // or .dll / .dylib based on platform

    LLMBackend backend = {};
    LibHandle handle = nullptr;
    if (!load_llm_backend(libPath, backend, handle)) {
        std::cerr << "Failed to load backend: " << libPath << "\n";
        return 1;
    }

    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    LLM* llm = backend.LLM_Construct(command.c_str());
    backend.LLM_StartServer(llm);
    backend.LLM_Start(llm);
    return 0;
}
