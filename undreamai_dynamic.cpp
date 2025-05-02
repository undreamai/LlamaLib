#include "dynamic_loader.h"
#include <iostream>

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << "\n" \
                      << "File: " << __FILE__ << "\n" \
                      << "Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)

char* GetFromStringWrapper(StringWrapper* stringWrapper){
    int bufferSize(stringWrapper->GetStringSize());
    char* content = new char[bufferSize];
    stringWrapper->GetString(content, bufferSize);
    return content;
}

// Trim from the start (left trim)
std::string ltrim(const std::string &s) {
    std::string result = s;
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return result;
}

// Trim from the end (right trim)
std::string rtrim(const std::string &s) {
    std::string result = s;
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());
    return result;
}

// Trim from both ends (left & right trim)
std::string trim(const std::string &s) {
    return ltrim(rtrim(s));
}

std::string concatenate_streaming_result(std::string input)
{
    std::vector<std::string> contents;
    std::istringstream stream(input);
    std::string line;

    std::string output = "";
    while (std::getline(stream, line)) {
        if (line.find("data: ") == 0) {
            std::string json_str = line.substr(6);
            try {
                json parsed = json::parse(json_str);
                output += parsed["content"];
            } catch (const json::exception& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
            }
        }
    }
    return output;
}





int main(int argc, char ** argv) {
    std::string libPath = "libundreamai_avx2.so"; // or .dll / .dylib based on platform
    LLMBackend backend = {};
    LibHandle handle = nullptr;

    if (!load_llm_backend(libPath, backend, handle)) {
        std::cerr << "Failed to load backend: " << libPath << "\n";
        return 1;
    }


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

    std::cout<<"******* backend.LLM_Construct *******"<<std::endl;
    llm = backend.LLM_Construct(command.c_str());

    std::thread t([&]() {backend.LLM_Start(llm);return 1;});
    std::cout<<"******* backend.LLM_Started *******"<<std::endl;
    while(!backend.LLM_Started(llm)){}
    
    std::cout<<"******* backend.LLM_SetTemplate *******"<<std::endl;
    backend.LLM_SetTemplate(llm, "mistral");
    assert(llm->chatTemplate == "mistral");

    
    std::cout<<"******* backend.LLM_Tokenize *******"<<std::endl;
    data["content"] = prompt;
    backend.LLM_Tokenize(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout<<"******* backend.LLM_Detokenize *******"<<std::endl;
    backend.LLM_Detokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);

    std::cout<<"******* backend.LLM_Completion *******"<<std::endl;
    data.clear();
    data["id_slot"] = id_slot;
    data["prompt"] = prompt;
    data["cache_prompt"] = true;
    data["stream"] = false;
    data["n_predict"] = 50;
    data["n_keep"] = 30;
    backend.LLM_Completion(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("content") > 0);

    data["prompt"] = prompt + std::string(reply_data["content"]);

    backend.LLM_Tokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    std::cout << std::abs((float)data["n_predict"] - reply_data["tokens"].size()) << std::endl;
    ASSERT(std::abs((float)data["n_predict"] - reply_data["tokens"].size()) < 4);

    std::cout<<"******* backend.LLM_Completion 2 *******"<<std::endl;
    data["stream"] = true;
    backend.LLM_Completion(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = concatenate_streaming_result(reply);
    ASSERT(reply_data != "");

    std::cout<<"******* backend.LLM_Embeddings *******"<<std::endl;
    data["content"] = prompt;
    backend.LLM_Embeddings(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data["embedding"].size() == backend.LLM_Embedding_Size(llm));

    std::cout<<"******* backend.LLM_Lora_List *******"<<std::endl;
    backend.LLM_Lora_List(llm, stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);

    std::cout<<"******* backend.LLM_Cancel *******"<<std::endl;
    backend.LLM_Cancel(llm, id_slot);

    std::cout<<"******* backend.LLM_StopServer *******"<<std::endl;
    backend.LLM_StopServer(llm);
    std::cout<<"******* backend.LLM_Stop *******"<<std::endl;
    backend.LLM_Stop(llm);
    t.join();
    std::cout<<"******* backend.LLM_Delete *******"<<std::endl;
    backend.LLM_Delete(llm);


    unload_llm_backend(handle);
    return 0;
}
