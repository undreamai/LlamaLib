#include "LLM_service.h"
#include "LLM_client.h"
#ifdef RUNTIME_TESTS
#include "LLM_runtime.h"
#endif

#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#endif

#include <iostream>

std::string PROMPT = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
int ID_SLOT = 0;
int EMBEDDING_SIZE;

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

void test_tokenization(LLM* llm) {
    std::cout << "LLM_Tokenize" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    reply = std::string(LLM_Tokenize(llm, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data.count("tokens") > 0);
    ASSERT(reply_data["tokens"].size() > 0);

    std::cout << "LLM_Detokenize" << std::endl;
    reply = std::string(LLM_Detokenize(llm, reply.c_str()));
    reply_data = json::parse(reply);
    ASSERT(trim(reply_data["content"]) == data["content"]);
}

int counter = 0;

void count_calls(const char* c)
{
    counter++;
}

void test_completion(LLM* llm, bool stream) {
    std::cout << "LLM_Completion ( ";
    if (!stream) std::cout << "no ";
    std::cout << "streaming )" << std::endl;

    json data;
    std::string reply;

    data["id_slot"] = ID_SLOT;
    data["prompt"] = PROMPT;
    data["cache_prompt"] = true;
    data["n_predict"] = 10;
    data["n_keep"] = 30;
    data["stream"] = stream;

    counter = 0;
    reply = std::string(LLM_Completion(llm, data.dump().c_str(), static_cast<CharArrayFn>(count_calls)));

    std::string reply_data;
    if (stream)
    {
        ASSERT(counter > 0);
        reply_data = concatenate_streaming_result(reply);
    }
    else
    {
        reply_data = json::parse(reply)["content"];
    }
    ASSERT(reply_data != "");
}

void test_embedding(LLM* llm) {
    std::cout << "LLM_Embeddings" << std::endl;
    json data, reply_data;
    std::string reply;

    data["content"] = PROMPT;
    reply = std::string(LLM_Embeddings(llm, data.dump().c_str()));
    reply_data = json::parse(reply);

    ASSERT(reply_data["embedding"].size() == EMBEDDING_SIZE);
}

void test_lora(LLMProvider* llm) {
    std::cout << "LLM_Lora_List" << std::endl;
    std::string reply;
    reply = std::string(LLM_Lora_List(llm));
    json reply_data = json::parse(reply);
    ASSERT(reply_data.size() == 0);
}

void test_cancel(LLMWithSlot* llm) {
    std::cout << "LLM_Cancel" << std::endl;
    LLM_Cancel(llm, ID_SLOT);
}

void test_slot_save_restore(LLMWithSlot* llm) {
    std::cout << "LLM_Slot Save" << std::endl;
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
    if (getcwd(buffer, sizeof(buffer)) != nullptr)
        data["filepath"] = std::string(buffer) + "/" + filename;
    else
        data["filepath"] = filename;
#endif

    reply = std::string(LLM_Slot(llm, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    int n_saved = reply_data["n_saved"];
    ASSERT(n_saved > 0);

    std::ifstream f(filename);
    ASSERT(f.good());
    f.close();

    std::cout << "LLM_Slot Restore" << std::endl;
    data["action"] = "restore";
    reply = std::string(LLM_Slot(llm, data.dump().c_str()));
    reply_data = json::parse(reply);
    ASSERT(reply_data["filename"] == filename);
    ASSERT(reply_data["n_restored"] == n_saved);

    std::remove(filename.c_str());
}

LLMService* start_llm_service(const std::string& command)
{
    LLMService* llm_service = LLM_Construct(command.c_str());
    LLM_Start(llm_service);
    return llm_service;
}

#ifdef RUNTIME_TESTS
LLMRuntime* start_llm_lib(std::string command)
{
    LLMRuntime* llmlib = LLMRuntime_Construct(command);
    if (!llmlib) {
        std::cerr << "Failed to load any backend." << std::endl;
        return nullptr;
    }
    LLM_Start(llmlib);
    LLM_Start_Server(llmlib);
    return llmlib;
}
#endif

void stop_llm_service(LLMProvider* llm)
{
    std::cout << "LLM_Stop_Server" << std::endl;
    LLM_Stop_Server(llm);
    std::cout << "LLM_Stop" << std::endl;
    LLM_Stop(llm);
    std::cout << "LLM_Delete" << std::endl;
    LLM_Delete(llm);
}

class TestLLM : public LLMProvider {
public:
    std::vector<int> TOKENS = std::vector<int>{1, 2, 3};
    std::string CONTENT = "my message";
    std::vector<float> EMBEDDING = std::vector<float>{0.1f, 0.2f, 0.3f};
    std::vector<LoraIdScalePath> LORAS = {
        {1, 1.0f, "model1.lora"},
        {2, 0.5f, "model2.lora"}
    };
    std::string SAVE_PATH = "test.save";

    std::string handle_tokenize_impl(const json& data) override {
        json result;
        result["tokens"] = TOKENS;
        return result.dump();
    }

    std::string handle_detokenize_impl(const json& data) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string handle_embeddings_impl(const json& data, httplib::Response*, std::function<bool()>) override {
        json result;
        result["embedding"] = EMBEDDING;
        return result.dump();
    }

    std::string handle_completions_impl(const json& data, CharArrayFn, httplib::Response*, std::function<bool()>, int) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string handle_slots_action_impl(const json& data, httplib::Response*) override {
        json result;
        result["filename"] = SAVE_PATH;
        return result.dump();
    }

    void handle_cancel_action_impl(int id_slot) override {
        cancel_called = true;
        cancelled_slot = id_slot;
    }

    std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response*) override {
        json result;
        result["success"] = true;
        return result.dump();
    }

    std::string handle_lora_adapters_list_impl() override {
        json j = json::array();
        for (auto& l : LORAS)
            j.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});
        return j.dump();
    }

    bool cancel_called = false;
    int cancelled_slot = -1;

    int get_status() override { return 0; }
    std::string get_status_message() override { return ""; }
    void start_server() override { }
    void stop_server() override { }
    void join_server() override { }
    void start_service() override { }
    void stop_service() override { }
    void join_service() override { }
    void set_SSL(const char* SSL_cert, const char* SSL_key) override { }
    bool is_running() override { return true; }
    int embedding_size() override { return 0; }
};

void run_mock_tests() {
    TestLLM llm;

    // --- Tokenize ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"tokens", llm.TOKENS}};

        ASSERT(llm.build_tokenize_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_tokenize_json(output_json) == llm.TOKENS);
        ASSERT(json::parse(llm.handle_tokenize_json(llm.CONTENT))["tokens"] == llm.TOKENS);
        ASSERT(llm.handle_tokenize(llm.CONTENT.c_str()) == llm.TOKENS);
        ASSERT(llm.handle_tokenize(llm.CONTENT) == llm.TOKENS);
        ASSERT(llm.handle_tokenize(input_json) == llm.TOKENS);
        ASSERT(llm.handle_tokenize_json(llm.CONTENT.c_str()) == output_json.dump());
        ASSERT(llm.handle_tokenize_json(llm.CONTENT) == output_json.dump());
        ASSERT(llm.handle_tokenize_json(input_json) == output_json.dump());
    }

    // --- Detokenize ---
    {
        json input_json = {{"tokens", llm.TOKENS}};
        json output_json = {{"content", llm.CONTENT}};

        ASSERT(llm.build_detokenize_json(llm.TOKENS)["tokens"] == llm.TOKENS);
        ASSERT(llm.parse_detokenize_json(output_json) == llm.CONTENT);
        ASSERT(llm.handle_detokenize(llm.TOKENS) == llm.CONTENT);
        ASSERT(llm.handle_detokenize(input_json) == llm.CONTENT);
        ASSERT(llm.handle_detokenize_json(llm.TOKENS) == output_json.dump());
        ASSERT(llm.handle_detokenize_json(input_json) == output_json.dump());
    }

    // --- Embeddings ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"embedding", llm.EMBEDDING}};

        ASSERT(llm.build_embeddings_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_embeddings_json(output_json) == llm.EMBEDDING);
        ASSERT(json::parse(llm.handle_embeddings_json(llm.CONTENT))["embedding"] == llm.EMBEDDING);
        ASSERT(json::parse(llm.handle_embeddings_json(llm.CONTENT.c_str()))["embedding"] == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(llm.CONTENT) == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(llm.CONTENT.c_str()) == llm.EMBEDDING);
        ASSERT(llm.handle_embeddings(input_json) == llm.EMBEDDING);
    }

    // --- Completions ---
    {
        int id_slot = 1;
        json params = {{"temp", 0.7}};
        json input_json = {
            {"prompt", llm.CONTENT},
            {"id_slot", id_slot},
            {"temp", 0.7}
        };
        json output_json = {{"content", llm.CONTENT}};

        ASSERT(llm.build_completions_json(llm.CONTENT, id_slot, params) == input_json);
        ASSERT(llm.parse_completions_json(output_json) == llm.CONTENT);
        ASSERT(llm.handle_completions(llm.CONTENT, id_slot, params) == llm.CONTENT);
        ASSERT(llm.handle_completions(input_json) == llm.CONTENT);
        ASSERT(llm.handle_completions_json(llm.CONTENT, id_slot, params) == output_json.dump());
    }

    // --- Slots Action ---
    {
        int id_slot = 42;
        std::string action = "load";
        json input_json = {
            {"id_slot", id_slot},
            {"action", action},
            {"filepath", llm.SAVE_PATH}
        };
        json output_json = {{"filename", llm.SAVE_PATH}};

        ASSERT(llm.build_slots_action_json(id_slot, action, llm.SAVE_PATH) == input_json);
        ASSERT(llm.parse_slots_action_json(output_json) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action(id_slot, action, llm.SAVE_PATH) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action(input_json) == llm.SAVE_PATH);
        ASSERT(llm.handle_slots_action_json(id_slot, action, llm.SAVE_PATH) == output_json.dump());
    }

    // --- LoRA Apply ---
    {
        std::vector<LoraIdScale> loras;
        for (auto& l : llm.LORAS)
            loras.push_back({l.id, l.scale});

        json input_json = json::array();
        for (const auto& l : loras)
            input_json.push_back({{"id", l.id}, {"scale", l.scale}});
        json output_json = {{"success", true}};

        ASSERT(llm.build_lora_adapters_apply_json(loras) == input_json);
        ASSERT(llm.parse_lora_adapters_apply_json(output_json));
        ASSERT(llm.handle_lora_adapters_apply_json(loras) == output_json.dump());
        ASSERT(llm.handle_lora_adapters_apply(input_json));
        ASSERT(llm.handle_lora_adapters_apply(loras));
    }

    // --- LoRA List ---
    {
        json input_json = json::array();
        for (const auto& l : llm.LORAS)
            input_json.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});

        ASSERT(llm.parse_lora_adapters_list_json(input_json) == llm.LORAS);
        ASSERT(llm.handle_lora_adapters_list() == llm.LORAS);
    }
}

void run_LLM_tests(LLM* llm)
{
    test_tokenization(llm);
    test_completion(llm, false);
    test_completion(llm, true);
    test_embedding(llm);
}

void run_LLMWithSlot_tests(LLMWithSlot* llm)
{
    run_LLM_tests(llm);
    test_cancel(llm);
    test_slot_save_restore(llm);
}

void run_LLMProvider_tests(LLMProvider* llm)
{
    run_LLMWithSlot_tests(llm);
    test_lora(llm);
}

void set_SSL(LLMProvider* llm, RemoteLLMClient* llm_remote_client)
{
    std::string server_crt = "-----BEGIN CERTIFICATE-----\n"
"MIIDJDCCAgwCFCurq2NTlmOrL3EBdiAnx2+x9YkqMA0GCSqGSIb3DQEBCwUAMEwx\n"
"CzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTENMAsGA1UEBwwEVGVzdDEQMA4GA1UE\n"
"CgwHVGVzdE9yZzEPMA0GA1UEAwwGVGVzdENBMCAXDTI1MDYxMDEyMDUzMVoYDzIx\n"
"MjUwNTE3MTIwNTMxWjBPMQswCQYDVQQGEwJVUzELMAkGA1UECAwCQ0ExDTALBgNV\n"
"BAcMBFRlc3QxEDAOBgNVBAoMB1Rlc3RPcmcxEjAQBgNVBAMMCWxvY2FsaG9zdDCC\n"
"ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAOHpa/VB/m9DN2QuGltcxT6p\n"
"ltT31RMCBaWILjltgoEeRZ1G985ToiemSUaeiA+b51vGjwUcp6yz6fHUgkll/LF0\n"
"ouOsiGMt6BT4wbp05RRdaj4telMju7ioWuBZKxAJ1CCJcipreA0Gk1gwf00bNWm2\n"
"FR8zMd1PjnDfsZ24iw7UnOL2MNKUfhrZVFK4nuqS+4WsXnGOX0iPsz+9xr1Y2M9Z\n"
"QVpvTlkX4wAw6FOfwoP1u3mrnk9wuyeD+cxFZj+drFMj+6i4WY4T/0iJbMXk11ng\n"
"ojAI65BZ+5SIBCQZPFN/Acta/8e9XpTYYnNDGc+ahHtxIvjkC6mMh7tpZ8xM7W0C\n"
"AwEAATANBgkqhkiG9w0BAQsFAAOCAQEAIpAcWnHELZ083sUFjX5wNxpEgmv5zk8X\n"
"0tJ6ZkyoxnKthp6ZzTF+aOP0kaeNdB5fp/DmEDBeeJVWaSCTpZs5H5oNZ6Mpdk9v\n"
"3bhRB66KhfRwHEm3FcNG8L84vK4cjPIB0/DkO7RtT/8KedrDlCIg/otLaBY8YVhp\n"
"7nDXqoKszC4ed2LLUO73IOIQUXwzFTmRmxKkvhNwjIwX6bMPqdJwdT9+/4592Nz6\n"
"aU6H9ejOxiJwT8KqkSBMs+KaTcrR5KU9O56pM+8NZ4jeJwtA/r0ER9KmfbeL6+mW\n"
"QXhRKl9XsXGQPwcMsB17qgbuZLgOPNhW4Jbzh+1BWMp1gnTKHqT5dw==\n"
"-----END CERTIFICATE-----\n";

    std::string server_key = "-----BEGIN PRIVATE KEY-----\n"
"MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDh6Wv1Qf5vQzdk\n"
"LhpbXMU+qZbU99UTAgWliC45bYKBHkWdRvfOU6InpklGnogPm+dbxo8FHKess+nx\n"
"1IJJZfyxdKLjrIhjLegU+MG6dOUUXWo+LXpTI7u4qFrgWSsQCdQgiXIqa3gNBpNY\n"
"MH9NGzVpthUfMzHdT45w37GduIsO1Jzi9jDSlH4a2VRSuJ7qkvuFrF5xjl9Ij7M/\n"
"vca9WNjPWUFab05ZF+MAMOhTn8KD9bt5q55PcLsng/nMRWY/naxTI/uouFmOE/9I\n"
"iWzF5NdZ4KIwCOuQWfuUiAQkGTxTfwHLWv/HvV6U2GJzQxnPmoR7cSL45AupjIe7\n"
"aWfMTO1tAgMBAAECggEABpl1jGgdoS1zAEuyfGnE31Q/8j+9Kz17Yb8NLqNK1S/H\n"
"s9T/ZzkdOxBKArSd3+rbgtxVkD4qjcqBso1VMwS2MY7pNUJ0h4UvSvGLY0GH8aTa\n"
"9i8I7EXWdYoBgZ1JO0I2Pq8VNTUHgEXpZwGfrmZ1lH17t3oc4kyxKg3219csxMWW\n"
"Lz6eeKXzDYzssJ9acb6skeSYuMOfjpBYIGo+5ebENntVSrpV2dzQsVNnwe79eAXt\n"
"vFdphRfXtx3sZ4tq4toEvTRZfM/R8/RS4ZeQixthJU7KFGWUVrJ4bgJmCijhD1mG\n"
"Dg79v19u2837Ve7AGY+PGNQ3USzx/KDJpolCqtowRwKBgQDjjoCI2K5iGI08Zr3E\n"
"tAxcLTpJcw7M5r6r27IXpvyATSMAhzTbrrWsgBFaoOsjSio/KnYMSTXnZJGUuTpi\n"
"S+LxsoF4VT893Sncuz5H56/N4gD0qtzPw9DQkzNWiUeWH3p8xwnAZHt1laXcwSu5\n"
"m+dDhvY0/xs5fXFRA8tlDPOPkwKBgQD+JklvHGiF2HJwGfj4r+ysPXrCG82NaNmP\n"
"rhMhVAyAdOanx09/aLToub3t7bVNuU6gcY35uH36elH/ydtr8onpEGjsI1xj/rNV\n"
"Tv0rVI51nqkAu9dNCVXe7Fw5/8kg/be2WOMGPf5JG27PQEmD3NSHK8wCUKo+vXx0\n"
"hIyq0Fmu/wKBgCfCj2zZx2Z2eb8TCJdlCj/U2zlYND7TFn+6zFxbngTg9XuzJCY6\n"
"WZ4BZobaVRt+avFMfwHYjOWYaeN9ldj0/3tRwFOBOaKakSTzRoeT0OD9W0Nk014u\n"
"Db9T6QV2yR5O87z3nhmStQuvkSKIUhaFShw/aaeK53vdEj6glhpa7/enAoGBAJqH\n"
"XQ8aDtOTD8HpiOBs11LC7uknTow0vFQIW8lf+VoBul05arTlTVpT1Y/dgOeJTK1x\n"
"XgoAi1jJFyKX8bpo9kGnoKQzu/Fw5Elyhaza9OO/XLL9g6NrkbLBtDHvvLM6kYFl\n"
"+mPJPdvlujJ5vDlZBEBL+PdPZLRRMmMGVSFnHaCxAoGAZYpfmWE3fK4PxZOv6vGc\n"
"ceAT8fdUwZbGVfY6xnaix7w3w1MrrGlbsaW/XOByVIkcQcQin9uK1MH63rrI1nK5\n"
"I19MOxy2t8of6Ey4isTgimH3hgAQiBufl/md+LqQqPqL4H4u7rC1fo0ZN6ccP/yn\n"
"R/AsCNV9+q4bcl6sSWio8nY=\n"
"-----END PRIVATE KEY-----\n";

    std::string client_crt = "-----BEGIN CERTIFICATE-----\n"
"MIIDezCCAmOgAwIBAgIUbXjhnPAG+qjvIgVocERmAYbctdswDQYJKoZIhvcNAQEL\n"
"BQAwTDELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAkNBMQ0wCwYDVQQHDARUZXN0MRAw\n"
"DgYDVQQKDAdUZXN0T3JnMQ8wDQYDVQQDDAZUZXN0Q0EwIBcNMjUwNjEwMTIwNTIz\n"
"WhgPMjEyNTA1MTcxMjA1MjNaMEwxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTEN\n"
"MAsGA1UEBwwEVGVzdDEQMA4GA1UECgwHVGVzdE9yZzEPMA0GA1UEAwwGVGVzdENB\n"
"MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArTEY9T62b4IkA2RXI8f3\n"
"oDQmG/Katspxo2mOGr4g5nmjBLUzJwsCoqdGzkSQvhpv9rviJe86cb/JJieXJImL\n"
"46I9lrDnDBuPbsAxob/NH8J+PJidVb/L80Bry+Mhh1bEBn8fANkK6/ipNFCrc5wa\n"
"8Mw99DiI7hJgfRopV9w5F8ZKuA+f8oXcraboI+k/Y2Tsl11anNaeFuOlkuzNu2Vn\n"
"sgvirF1q9EylLV+6XdlSE27HQmNeZtTHv1lJqqICsuEtXISBFZ+5XCJpNNOJr/iW\n"
"4ZpTgZ/mDmBiD8x5YR+kPeDegGo2L9e97WeFhL0FCiCWEQU4GZgtuemBJCOirgdI\n"
"4wIDAQABo1MwUTAdBgNVHQ4EFgQUgL8V38xcwegVy92cQIRGSkjPEhgwHwYDVR0j\n"
"BBgwFoAUgL8V38xcwegVy92cQIRGSkjPEhgwDwYDVR0TAQH/BAUwAwEB/zANBgkq\n"
"hkiG9w0BAQsFAAOCAQEAgqf122GLRTjXVnjg5T3NvPe+2EpFgq46zEO6SQP/DX/8\n"
"JI1rQzq3BWAZkWJn3UP/lEAiHqiSrcdEyI8j3iXx65GcpJg/slN+IkSsHpzh75El\n"
"SCPEcyMVPx76D8v3dV9V4YZbt3/2fUmsP8YNkNRMeaL6+d2wgB+wAnSRskb2ywag\n"
"saxATZAcfLhevVH+BUiT0N08AdqtjzuILIviMdZSc7LEkK8Jut4hph4MDgYRK6O/\n"
"n2zEmM8DCyuiNG5y+i+bZy8GMULqcaUtTMqe9M7K6wowtCrZRG6a1LSSTIwiGQS7\n"
"n8CVEjaTnlwiTA93kzQF6bWZgTUe3QDBNTpFSVAywg==\n"
"-----END CERTIFICATE-----\n";

    LLM_SetSSL(llm, server_crt.c_str(), server_key.c_str());
    llm_remote_client->set_SSL(client_crt.c_str());
}

int main(int argc, char** argv) {
    SetDebugLevel(ERR);

    run_mock_tests();

    std::string command = args_to_command(argc, argv);

    LLMService* llm_service = start_llm_service(command);
    EMBEDDING_SIZE = LLM_Embedding_Size(llm_service);

    std::cout << "-------- LLM service --------" << std::endl;
    run_LLMProvider_tests(llm_service);

    std::cout << "-------- LLM client --------" << std::endl;
    LLMClient llm_client(llm_service);
    run_LLMWithSlot_tests(&llm_client);

    std::cout << "-------- LLM remote client --------" << std::endl;
    RemoteLLMClient llm_remote_client("https://localhost", 8080);
    set_SSL(llm_service, &llm_remote_client);
    LLM_Start_Server(llm_service);
    run_LLM_tests(&llm_remote_client);

    stop_llm_service(llm_service);

#ifdef RUNTIME_TESTS
    std::cout << "-------- LLM lib --------" << std::endl;
    LLMRuntime* llmlib = start_llm_lib(command);
    EMBEDDING_SIZE = LLM_Embedding_Size(llmlib);
    run_LLMProvider_tests(llmlib);
    stop_llm_service(llmlib);
#endif

    return 0;
}
