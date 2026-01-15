#include "LlamaLib.h"

int main(int argc, char **argv)
{
    int server_port = 13333;

    // create LLM
    LLMService* llm_server = LLMServiceBuilder().model("model.gguf").numGPULayers(10).build();
    // alternatively using the LLMService constructor:
    // LLMService* llm_server = new LLMService("model.gguf", 1, -1, 10);

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