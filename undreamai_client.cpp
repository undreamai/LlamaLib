#include "LLMClient.h"
#include <iostream>

int main(int argc, char** argv) {
	std::cout << "connecting" << std::endl;
	RemoteLLMClient client("http://localhost", 8000);
	std::vector<int> tokens = client.handle_tokenize("hello world");
	for (int i = 0; i < tokens.size(); i++)
		std::cout << tokens[i] << " ";
}