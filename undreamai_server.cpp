#include "undreamai.h"

int main(int argc, char ** argv) {
    int i = 1; // Start from 1 to skip the program name
    std::string templateValue = "";

    while (i < argc) {
        if (std::string(argv[i]) == "--template") {
            if (i + 1 < argc) {
                templateValue = argv[i + 1];

                // Shift the remaining arguments to remove --template and its value
                for (int j = i; j < argc - 2; ++j) {
                    argv[j] = argv[j + 2];
                }
                argc -= 2;
            } else {
                std::cerr << "Error: --template option requires a value" << std::endl;
                exit(1);
            }
        } else {
            ++i;
        }
    }

    LLM llm(argc, argv);
    if (templateValue != ""){
        std::cout<<"Using template "<<templateValue<<std::endl;
        llm.chatTemplate = templateValue;
    }
    llm.start_server();
    llm.start_service();
}
