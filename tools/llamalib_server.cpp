#include "LLM_runtime.h"

int main(int argc, char **argv)
{
    std::string command = args_to_command(argc, argv);

    LLMRuntime *llm = LLMRuntime::from_command(command.c_str());
    if (!llm)
    {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }

    LLM_Debug(1);
    LLM_Start(llm);
    LLM_Start_Server(llm);
    LLM_Join_Server(llm);

    return 0;
}
