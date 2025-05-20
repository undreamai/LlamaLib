#include "undreamai.h"
#include "LLMClient.h"
#include <iostream>


int main(int argc, char** argv) {
	//std::cout << "connecting" << std::endl;
	//LLMClient client("http://localhost", 8000);
	//std::vector<int> tokens = client.handle_tokenize("hello world");
	//for (int i = 0; i < tokens.size(); i++)
	//	std::cout << tokens[i] << " ";


	LLM* llm;
	StringWrapper* stringWrapper = StringWrapper_Construct();
	std::string prompt = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
	std::string command = "";
	for (int i = 1; i < argc; ++i) {
		command += argv[i];
		if (i < argc - 1) command += " ";
	}
	json data;
	json reply_data;
	std::string reply;
	int id_slot = 0;

	std::cout << "******* LLM_Construct *******" << std::endl;
	llm = LLM_Construct(command.c_str());

	std::thread t([&]() {LLM_Start(llm);return 1;});
	std::cout << "******* LLM_Started *******" << std::endl;
	while (!LLM_Started(llm)) {}

	std::cout << "******* LLM_Tokenize *******" << std::endl;
	data["content"] = prompt;


	// Create an LLM client with just the LLM pointer
	LLMClient client(llm);

	//// Set the specific function pointers you need
	//client.setFunctionPointer("LLM_Tokenize", (void*)&LLM_Tokenize);

	//// Now these functions can be called successfully
	std::cout << client.handle_tokenize_json(data) << std::endl;
}