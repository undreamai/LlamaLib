#include "undreamai.h"

int main(int argc, char ** argv) {
    int i = 1;
    std::string templateValue = "";
    std::string command = "";

    while (i < argc) {
        if (std::string(argv[i]) == "--template") {
            if (i + 1 < argc) {
                templateValue = argv[i + 1];
                i++;
            } else {
                std::cerr << "Error: --template option requires a value" << std::endl;
                exit(1);
            }
        } else {
            command += argv[i];
            if (i < argc - 1) command += " ";
        }
        i++;
    }

    LLM* llm = LLM_Construct(command.c_str());
    if (templateValue != ""){
        std::cout<<"Using template "<<templateValue<<std::endl;
        LLM_SetTemplate(llm, templateValue.c_str());
    }
    LLM_StartServer(llm);
    LLM_Start(llm);
    return 0;
}
