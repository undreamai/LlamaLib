#include "LlamaLib.h"
#include <iostream>

int main(int argc, char **argv)
{
    // create embedding LLM
    LLMService* llm_service_embeddings = LLMServiceBuilder().model("model.gguf").embeddingOnly(true).build();
    // alternatively using the LLMService constructor:
    // LLMService* llm_service_embeddings = new LLMService("model.gguf", 1, -1, 0, false, 4096, 2048, true);
    llm_service_embeddings->start();

    std::cout << std::endl << "----------------------- embeddings -----------------------" << std::endl;
    std::vector<float> embeddings = llm_service_embeddings->embeddings("this is the text I want to embed");
    std::cout << "embeddings: ";
    size_t maxCount = std::min<size_t>(embeddings.size(), 25);
    for (size_t i = 0; i < maxCount; ++i) std::cout << embeddings[i] << " ";
    std::cout << "..." << std::endl;

    delete llm_service_embeddings;

    return 0;
}
