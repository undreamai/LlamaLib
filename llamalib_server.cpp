#ifdef LLAMALIB_BUILD_RUNTIME_LIB
#include "LLM_lib.h"
#else
#include "LLM_service.h"
#endif

int main(int argc, char ** argv) {
    std::string command = args_to_command(argc, argv);
    std::cout << command << std::endl;

#ifdef LLAMALIB_BUILD_RUNTIME_LIB
    LLMLib* llm = LLMLib_Construct(command);
    if (!llm) {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }
#else
    LLMService* llm = LLM_Construct(command.c_str());
#endif
    LLM_Start(llm);
    LLM_Start_Server(llm);
    LLM_Join_Server(llm);

    return 0;
}
