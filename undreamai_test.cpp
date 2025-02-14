#include "undreamai.h"

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

    std::cout<<"******* LLM_Construct *******"<<std::endl;
    llm = LLM_Construct(command.c_str());

    std::thread t([&]() {LLM_Start(llm);return 1;});
    std::cout<<"******* LLM_Started *******"<<std::endl;
    while(!LLM_Started(llm)){}
    
    std::cout<<"******* LLM_SetTemplate *******"<<std::endl;
    LLM_SetTemplate(llm, "mistral");
    assert(llm->chatTemplate == "mistral");

    
    std::cout<<"******* LLM_Tokenize *******"<<std::endl;
    data["content"] = prompt;
    LLM_Tokenize(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout<<"******* LLM_Detokenize *******"<<std::endl;
    LLM_Detokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);

    std::cout<<"******* LLM_Completion *******"<<std::endl;
    data.clear();
    data["id_slot"] = id_slot;
    data["prompt"] = prompt;
    data["cache_prompt"] = true;
    data["stream"] = false;
    data["n_predict"] = 50;
    data["n_keep"] = 30;
    LLM_Completion(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("content") > 0);

    data["prompt"] = prompt + std::string(reply_data["content"]);

    LLM_Tokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    std::cout << std::abs((float)data["n_predict"] - reply_data["tokens"].size()) << std::endl;
    ASSERT(std::abs((float)data["n_predict"] - reply_data["tokens"].size()) < 4);

    std::cout<<"******* LLM_Completion 2 *******"<<std::endl;
    data["stream"] = true;
    LLM_Completion(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = concatenate_streaming_result(reply);
    ASSERT(reply_data != "");

    std::cout<<"******* LLM_Embeddings *******"<<std::endl;
    data["content"] = prompt;
    LLM_Embeddings(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data["embedding"].size() == LLM_Embedding_Size(llm));

    std::cout<<"******* LLM_Lora_List *******"<<std::endl;
    LLM_Lora_List(llm, stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);

    std::cout<<"******* LLM_Cancel *******"<<std::endl;
    LLM_Cancel(llm, id_slot);

    /*
    std::string filename = "test_undreamai.save";
    data.clear();
    data["id_slot"] = id_slot;
    data["action"] = "save";
    data["filename"] = filename;
    LLM_Slot(llm, data.dump().c_str());
    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    data["action"] = "restore";
    LLM_Slot(llm, data.dump().c_str());
    std::remove(filename.c_str());
    */

    std::cout<<"******* LLM_StopServer *******"<<std::endl;
    LLM_StopServer(llm);
    std::cout<<"******* LLM_Stop *******"<<std::endl;
    LLM_Stop(llm);
    t.join();
    std::cout<<"******* LLM_Delete *******"<<std::endl;
    LLM_Delete(llm);

    return 0;
}
