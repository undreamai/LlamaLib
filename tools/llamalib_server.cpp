#include "LlamaLib.h"

int main(int argc, char **argv)
{
    std::string command = args_to_command(argc, argv);

    LLMService *llm = LLMService::from_command(command.c_str());
    if (!llm)
    {
        std::cout << "Failed to load any backend." << std::endl;
        return 1;
    }

    llm->debug(1);
    llm->start();
    llm->start_server();
    std::cout << "service started" << std::endl;
    llm->join_server();

    return 0;
}
