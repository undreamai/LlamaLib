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

/*
int main(int argc, char ** argv) {
    LLM llm(argc, argv);
    llm.start_server();
    llm.start_service();
}
*/

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

    llm = LLM_Construct(command.c_str());

    LLM_StartServer(llm);
    std::thread t([&]() {LLM_Start(llm);return 1;});
    
    LLM_SetTemplate(llm, "mistral");
    assert(llm->chatTemplate == "mistral");
    
    data["content"] = prompt;
    LLM_Tokenize(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    LLM_Detokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);

    data.clear();
    data["prompt"] = prompt;
    data["stream"] = false;
    data["n_predict"] = 12;
    LLM_Completion(llm, data.dump().c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("content") > 0);

    LLM_Tokenize(llm, reply.c_str(), stringWrapper);
    reply = GetFromStringWrapper(stringWrapper);
    reply_data = json::parse(reply);
    ASSERT(data["n_predict"] == reply_data["tokens"].size());

    std::string filename = "test_undreamai.save";
    data.clear();
    data["action"] = "save";
    data["filename"] = filename;
    LLM_Slot(llm, data.dump().c_str());
    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    data["action"] = "restore";
    LLM_Slot(llm, data.dump().c_str());
    std::remove(filename.c_str());

    LLM_StopServer(llm);
    LLM_Stop(llm);
    t.join();
    LLM_Delete(llm);

    return 0;
}
