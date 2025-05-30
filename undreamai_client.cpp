#include "LLM_service.h"
#include "LLM_client.h"
#include <iostream>


int main(int argc, char** argv) {
	std::string prompt = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
	std::string command = "";
	for (int i = 1; i < argc; ++i) {
		command += argv[i];
		if (i < argc - 1) command += " ";
	}

	std::cout << "******* LLM_Construct *******" << std::endl;

	/*LLMService* llm_service = LLM_Construct(command.c_str());
	LLM_Start(llm_service);*/

	std::cout << "-------- LLM remote client --------" << std::endl;
	//LLM_StartServer(llm_service);
	RemoteLLMClient client("localhost", 8080);

	std::cout << "******* LLM_Tokenize *******" << std::endl;
	json data;
	data["content"] = prompt;
	std::cout << client.handle_tokenize_json(data) << std::endl;
}
