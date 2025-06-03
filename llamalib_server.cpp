
#ifdef LLAMALIB_BUILD_RUNTIME_LIB
#include "LLM_lib.h"
#else
#include "LLM_service.h"
#endif

int main(int argc, char ** argv) {
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

#ifdef LLAMALIB_BUILD_RUNTIME_LIB
    LLMLib* llm = Load_LLM_Library(command);
    if (!llm) {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }
    LLMLib_LLM_Start_Server(llm);
    LLMLib_LLM_Join_Server(llm);
#else
    LLMService* llm = LLM_Construct(command.c_str());
    LLM_Start(llm);
    LLM_Start_Server(llm);
    LLM_Join_Server(llm);
#endif

    return 0;
}
