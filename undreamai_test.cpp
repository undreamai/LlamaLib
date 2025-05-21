#include "LLM_service.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include <iostream>

int ID_SLOT = 0;

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << "\n" \
                      << "File: " << __FILE__ << "\n" \
                      << "Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)

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

void test_tokenization(LLMService* llm, StringWrapper* wrapper, const std::string& prompt) {
    std::cout << "******* LLM_Tokenize *******" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = prompt;
    LLM_Tokenize(llm, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout << "******* LLM_Detokenize *******" << std::endl;
    LLM_Detokenize(llm, reply.c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);
}

void test_completion(LLMService* llm, StringWrapper* wrapper, const std::string& prompt, bool stream) {
    std::cout << "******* LLM_Completion ( ";
    if (!stream) std::cout << "no ";
    std::cout << "streaming ) *******" << std::endl;
    json data;
    std::string reply;

    data["id_slot"] = ID_SLOT;
    data["prompt"] = prompt;
    data["cache_prompt"] = true;
    data["stream"] = false;
    data["n_predict"] = 10;
    data["n_keep"] = 30;
    data["stream"] = stream;

    LLM_Completion(llm, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    std::string reply_data;
    if (stream)
    {
        reply_data = concatenate_streaming_result(reply);
    }
    else {
        reply_data = json::parse(reply)["content"];
    }
    ASSERT(reply_data != "");
}

void test_embedding(LLMService* llm, StringWrapper* wrapper, const std::string& prompt) {
    std::cout << "******* LLM_Embeddings *******" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = prompt;
    LLM_Embeddings(llm, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);

    ASSERT(reply_data["embedding"].size() == LLM_Embedding_Size(llm));
}

void test_lora(LLMService* llm, StringWrapper* wrapper) {
    std::cout << "******* LLM_Lora_List *******" << std::endl;
    LLM_Lora_List(llm, wrapper);
    std::string reply = GetStringWrapperContent(wrapper);
    json reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);
}

void test_cancel(LLMService* llm) {
    std::cout << "******* LLM_Cancel *******" << std::endl;
    LLM_Cancel(llm, ID_SLOT);
}

void test_slot_save_restore(LLMService* llm, StringWrapper* wrapper) {
    std::cout << "******* LLM_Slot Save *******" << std::endl;
    std::string filename = "test_undreamai.save";
    json data;
    json reply_data;
    std::string reply;

    data["id_slot"] = ID_SLOT;
    data["action"] = "save";

#ifdef _WIN32
    char buffer[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, buffer);
    data["filepath"] = std::string(buffer) + "\\" + filename;
#else
    char buffer[PATH_MAX];
    getcwd(buffer, sizeof(buffer));
    data["filepath"] = std::string(buffer) + "/" + filename;
#endif
   
    LLM_Slot(llm, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    int n_saved = reply_data["n_saved"];
    ASSERT(n_saved > 0);

    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    std::cout << "******* LLM_Slot Restore *******" << std::endl;
    data["action"] = "restore";
    LLM_Slot(llm, data.dump().c_str(), wrapper);
    reply = GetStringWrapperContent(wrapper);
    std::cout << reply << std::endl;
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    ASSERT(reply_data["n_restored"] == n_saved);

    std::remove(filename.c_str());
}

LLMService* start_llm(const std::string& command)
{
    LLMService* llm = LLM_Construct(command.c_str());
    LLM_Start(llm);
    return llm;
}

void stop_llm(LLMService* llm)
{
    std::cout << "******* LLM_StopServer *******" << std::endl;
    LLM_StopServer(llm);
    std::cout << "******* LLM_Stop *******" << std::endl;
    LLM_Stop(llm);
    std::cout << "******* LLM_Delete *******" << std::endl;
    LLM_Delete(llm);
}

int main(int argc, char** argv) {
    std::string prompt = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
    std::string command;
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }

    StringWrapper* wrapper = StringWrapper_Construct();
    LLMService* llm = start_llm(command);

    test_tokenization(llm, wrapper, prompt);
    test_completion(llm, wrapper, prompt, false);
    test_completion(llm, wrapper, prompt, true);
    test_embedding(llm, wrapper, prompt);
    test_lora(llm, wrapper);
    test_cancel(llm);
    test_slot_save_restore(llm, wrapper);

    stop_llm(llm);
    delete wrapper;

    return 0;
}
