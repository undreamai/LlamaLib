#include "LLM_runtime.h"

int main(int argc, char ** argv) {
    std::string command = args_to_command(argc, argv);
    std::cout << command << std::endl;

    LLMRuntime* llm = LLMRuntime_Construct(command);
    if (!llm) {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }

    LLM_Start(llm);
    LLM_Start_Server(llm);
    LLM_Join_Server(llm);

    return 0;
}
