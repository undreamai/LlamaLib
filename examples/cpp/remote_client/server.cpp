#include "LlamaLib.h"

int main(int argc, char **argv)
{
    std::string model = "model.gguf";
    int server_port = 13333;

    std::cout << "Starting LLM server..." << std::endl;
    LLMService *llm_server = new LLMService(model);

    // show debug messages
    llm_server->debug(1);

    // start service and server
    llm_server->start();
    llm_server->start_server("", server_port);

    // wait until server exit
    llm_server->join_server();

    delete llm_server;
    return 0;
}