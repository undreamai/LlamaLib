
#include "LLM_client.h"
#ifdef RUNTIME_TESTS
#include "LLM_runtime.h"
#else
#include "LLM_service.h"
#endif
#include "LLM_agent.h"

#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#endif

#include <iostream>
#include <fstream>

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


int counter = 0;
std::string concat_result = "";
void count_calls(const char* c)
{
    concat_result += c;
    counter++;
}

void test_template(LLMProvider* llm, bool use_api) {
    std::string chat_template = "phi3";
    std::cout << "set_template" << std::endl;
    if(use_api) LLM_Set_Template(llm, chat_template.c_str());
    else llm->set_template(chat_template);

    std::cout << "get_template" << std::endl;
    std::string get_template;
    if(use_api) get_template = LLM_Get_Template(llm);
    else get_template = llm->get_template();
    ASSERT(get_template == chat_template);

    llm->set_template("chatml");
}

void test_apply_template(LLM* llm, bool use_api) {
    std::cout << "apply_template" << std::endl;
    json data = json::array();
    data.push_back({{"role", "user"}, {"content", "how are you?"}});
    data.push_back({{"role", "assistant"}, {"content", "fine, thanks, and you?"}});
    std::string data_formatted_gt = "<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\nfine, thanks, and you?";
    std::string data_formatted;
    if(use_api) data_formatted = LLM_Apply_Template(llm, data.dump().c_str());
    else data_formatted = llm->apply_template(data);
    ASSERT(data_formatted == data_formatted_gt);
}

void test_tokenize(LLM* llm, bool use_api) {
    std::cout << "tokenize" << std::endl;
    std::vector<int> tokens;
    if(use_api) {
        std::string reply = std::string(LLM_Tokenize(llm, PROMPT.c_str()));
        json reply_data = json::parse(reply);
        tokens = reply_data.get<std::vector<int>>();
    } else {
        tokens = llm->tokenize(PROMPT);
    }
    ASSERT(tokens.size() > 0);

    std::cout << "detokenize" << std::endl;
    std::string reply;
    if(use_api) {
        json tokens_json = tokens;
        reply = std::string(LLM_Detokenize(llm, tokens_json.dump().c_str()));
    } else {
        reply = llm->detokenize(tokens);
    }
    ASSERT(trim(reply) == PROMPT);
}

void test_completion(LLM* llm, bool stream, bool use_api) {
    std::cout << "completion ( ";
    if (!stream) std::cout << "no ";
    std::cout << "streaming )" << std::endl;

    counter = 0;
    concat_result = "";
    std::string reply;
    
    if(use_api) {
        if (stream) {
            reply = std::string(LLM_Completion(llm, PROMPT.c_str(), static_cast<CharArrayFn>(count_calls)));
        } else {
            reply = std::string(LLM_Completion(llm, PROMPT.c_str()));
        }
    } else {
        if (stream) {
            reply = llm->completion(PROMPT, static_cast<CharArrayFn>(count_calls));
        } else {
            reply = llm->completion(PROMPT);
        }
    }
    ASSERT(reply != "");
    if (stream)
    {
        ASSERT(counter > 3);
        ASSERT(reply == concat_result);
    }
}

void test_embeddings(LLM* llm, bool use_api) {
    std::cout << "embeddings" << std::endl;
    std::vector<float> embeddings;
    if(use_api) {
        std::string reply = std::string(LLM_Embeddings(llm, PROMPT.c_str()));
        embeddings = json::parse(reply).get<std::vector<float>>();
    } else {
        embeddings = llm->embeddings(PROMPT);
    }
    ASSERT(embeddings.size() == EMBEDDING_SIZE);
}

void test_lora_list(LLMProvider* llm, bool use_api) {
    std::cout << "lora_list" << std::endl;
    if(use_api) {
        json reply_data = json::parse(LLM_Lora_List(llm));
        ASSERT(reply_data.size() == 0);
    } else {
        std::vector<LoraIdScalePath> loras = llm->lora_list();
        ASSERT(loras.size() == 0);
    }
}

void test_cancel(LLMLocal* llm, bool use_api) {
    std::cout << "cancel" << std::endl;
    if(use_api) {
        LLM_Cancel(llm, ID_SLOT);
    } else {
        llm->cancel(ID_SLOT);
    }
}

void test_slot(LLMLocal* llm, bool use_api) {
    bool remote = false;
    if (LLMClient* client = dynamic_cast<LLMClient*>(llm)) {
        remote = client->is_remote();
    }

    std::cout << "slot Save" << std::endl;
    std::string filename = "test_undreamai.save";
    std::string filepath = filename;

#ifdef _WIN32
    char buffer[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, buffer);
    filepath = std::string(buffer) + "\\" + filename;
#else
    char buffer[PATH_MAX];
    if (getcwd(buffer, sizeof(buffer)) != nullptr)
        filepath = std::string(buffer) + "/" + filename;
#endif

    std::string reply;
    if(use_api) {
        reply = std::string(LLM_Save_Slot(llm, ID_SLOT, filepath.c_str()));
    } else {
        reply = llm->save_slot(ID_SLOT, filepath);
    }
    
    if (!remote) {
        ASSERT(reply == filename);
        std::ifstream f(filename);
        ASSERT(f.good());
        f.close();
    } else {
        ASSERT(reply == "");
    }

    std::cout << "slot Restore" << std::endl;
    if(use_api) {
        reply = std::string(LLM_Load_Slot(llm, ID_SLOT, filepath.c_str()));
    } else {
        reply = llm->load_slot(ID_SLOT, filepath);
    }
    
    if (!remote) {
        ASSERT(reply == filename);
        std::remove(filename.c_str());
    } else {
        ASSERT(reply == "");
    }
}

void test_agent_chat(LLMAgent* agent, bool stream, bool use_api) {
    std::cout << "agent chat ( ";
    if (!stream) std::cout << "no ";
    std::cout << "streaming )" << std::endl;

    std::string user_role = "random_person";
    std::string assistant_role = "random_bot";
    if(use_api) {
        agent->set_user_role(user_role);
        agent->set_assistant_role(assistant_role);
        ASSERT(agent->get_user_role() == user_role);
        ASSERT(agent->get_assistant_role() == assistant_role);
    } else {
        LLMAgent_Set_User_Role(agent, user_role.c_str());
        LLMAgent_Set_Assistant_Role(agent, assistant_role.c_str());
        ASSERT(LLMAgent_Get_User_Role(agent) == user_role);
        ASSERT(LLMAgent_Get_Assistant_Role(agent) == assistant_role);
    }

    std::string user_prompt = "Hello, how are you?";
    std::string reply;

    agent->clear_history();
    for (bool add_to_history: {true, false}) {
        counter = 0;
        concat_result = "";

        if(use_api) {
            if (stream) {
                reply = LLMAgent_Chat(agent, user_prompt.c_str(), add_to_history, static_cast<CharArrayFn>(count_calls));
            } else {
                reply = LLMAgent_Chat(agent, user_prompt.c_str(), add_to_history);
            }
        } else {
            if (stream) {
                reply = agent->chat(user_prompt, add_to_history, static_cast<CharArrayFn>(count_calls));
            } else {
                reply = agent->chat(user_prompt, add_to_history);
            }
        }

        ASSERT(reply != "");
        if (stream)
        {
            std::cout<<"counter: "<<counter<<std::endl;
            ASSERT(counter > 3);
            ASSERT(reply == concat_result);
        }

        size_t history_size;
        if(use_api) {
            history_size = LLMAgent_Get_History_Size(agent);
        } else {
            history_size = agent->get_history_size();
        }
        ASSERT(history_size == 3);

        json history = agent->get_history();
        ASSERT(history[0]["role"] == "system");
        ASSERT(history[1]["role"] == user_role);
        ASSERT(history[2]["role"] == assistant_role);
        ASSERT(history[0]["content"] == agent->get_system_prompt());
        ASSERT(history[1]["content"] == user_prompt);
    }

    if(use_api) {
        LLMAgent_Clear_History(agent);
    } else {
        agent->clear_history();
    }
}

void test_history(LLMAgent* agent, bool use_api) {
    std::cout << "History Management" << std::endl;

    // Test initial state
    std::string system_prompt ;
    if (use_api) system_prompt = LLMAgent_Get_System_Prompt(agent);
    else system_prompt = agent->get_system_prompt();
    ASSERT(!system_prompt.empty());

    // Test adding messages
    if (use_api) {
        LLMAgent_Add_Message(agent, "user", "Test user message");
        LLMAgent_Add_Message(agent, "assistant", "Test assistant response");
    } else {
        agent->add_message("user", "Test user message");
        agent->add_message("assistant", "Test assistant response");
    }

    size_t history_size;
    
    if (use_api) history_size = LLMAgent_Get_History_Size(agent);
    else history_size = agent->get_history_size();
    ASSERT(history_size == 3); // system + user + assistant

    // Test getting history as JSON
    json history;
    if (use_api) history = json::parse(LLMAgent_Get_History(agent));
    else history = agent->get_history();
    ASSERT(history.is_array());
    ASSERT(history.size() == 3);

    // test history formatting
    std::string data_formatted_gt = "<|im_start|>system\n"+system_prompt+"<|im_end|>\n<|im_start|>user\nTest user message<|im_end|>\n<|im_start|>assistant\nTest assistant response";
    std::string data_formatted;
    if(use_api) data_formatted = LLM_Apply_Template(agent, history.dump().c_str());
    else data_formatted = agent->apply_template(history);
    ASSERT(data_formatted == data_formatted_gt);

    // Test removing last message
    if (use_api) LLMAgent_Remove_Last_Message(agent);
    else agent->remove_last_message();
    ASSERT(agent->get_history_size() == 2);

    // Test clearing history
    if (use_api) LLMAgent_Clear_History(agent);
    else agent->clear_history();
    ASSERT(agent->get_history_size() == 1);

    // Create test history JSON
    json test_history = json::array();
    test_history.push_back({{"role", "system"}, {"content", "Test system prompt"}});
    test_history.push_back({{"role", "user"}, {"content", "Hello"}});
    test_history.push_back({{"role", "assistant"}, {"content", "Hi there!"}});
    test_history.push_back({{"role", "user"}, {"content", "How are you?"}});
    test_history.push_back({{"role", "assistant"}, {"content", "I'm doing well, thanks!"}});

    // Test set chat history
    if (use_api) LLMAgent_Set_History(agent, test_history.dump().c_str());
    else agent->set_history(test_history);
    ASSERT(agent->get_history_size() == 5);

    json retrieved_history;
    if (use_api) retrieved_history = json::parse(LLMAgent_Get_History(agent));
    else retrieved_history = agent->get_history();
    ASSERT(retrieved_history == test_history);

    // Test setting new system prompt
    std::string new_system_prompt = "You are a helpful test assistant.";
    if (use_api)
    {
        LLMAgent_Set_System_Prompt(agent, new_system_prompt.c_str());
        ASSERT(LLMAgent_Get_System_Prompt(agent) == new_system_prompt);
    } else {
        agent->set_system_prompt(new_system_prompt);
        ASSERT(agent->get_system_prompt() == new_system_prompt);
    }
    ASSERT(agent->get_history_size() == 1);
}

void test_save_history(LLMAgent* agent, bool use_api) {
    std::cout << "LLMAgent File Operations" << std::endl;

    // Add some test messages
    agent->add_message("user", "Test message 1");
    agent->add_message("assistant", "Test response 1");
    agent->add_message("user", "Test message 2");
    agent->add_message("assistant", "Test response 2");
    size_t original_size = agent->get_history_size();

    // Test saving to file
    std::string filename = "test_agent_history.json";

    if (use_api) LLMAgent_Save_History(agent, filename.c_str());
    else agent->save_history(filename);

    std::ifstream file(filename);
    ASSERT(file.good());
    file.close();

    agent->clear_history();
    ASSERT(agent->get_history_size() == 1);

    if (use_api) LLMAgent_Load_History(agent, filename.c_str());
    else agent->load_history(filename);
    ASSERT(agent->get_history_size() == original_size);

    std::remove(filename.c_str());
}

void test_ChatMessage() {
    std::cout << "ChatMessage Struct Tests" << std::endl;

    // Test constructors
    ChatMessage msg1;
    ASSERT(msg1.role.empty());
    ASSERT(msg1.content.empty());

    ChatMessage msg2("user", "Hello world");
    ASSERT(msg2.role == "user");
    ASSERT(msg2.content == "Hello world");

    // Test equality
    ChatMessage msg3("user", "Hello world");
    ASSERT(msg2 == msg3);

    ChatMessage msg4("assistant", "Hello world");
    ASSERT(!(msg2 == msg4));

    // Test JSON conversion
    json msg_json = msg2.to_json();
    ASSERT(msg_json["role"] == "user");
    ASSERT(msg_json["content"] == "Hello world");

    ChatMessage msg_from_json = ChatMessage::from_json(msg_json);
    ASSERT(msg_from_json == msg2);
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
    int cancelled_slot = -1;
    std::string chat_template = "";

    void start_server(const std::string& host="0.0.0.0", int port=0, const std::string& API_key="") override { }
    void stop_server() override { }
    void join_server() override { }
    void start() override { }
    void stop() override { }
    void join_service() override { }
    void set_SSL(const std::string& SSL_cert, const std::string& SSL_key) override { }
    bool started() override { return true; }
    int embedding_size() override { return 0; }
    int get_available_slot() override { return -1; }

    void debug(int debug_level) override {}
    void logging_callback(CharArrayFn callback) override {}

    std::string tokenize_json(const json& data) override {
        json result;
        result["tokens"] = TOKENS;
        return result.dump();
    }

    std::string detokenize_json(const json& data) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string embeddings_json(const json& data) override {
        json result;
        result["embedding"] = EMBEDDING;
        return result.dump();
    }

    std::string completion_json(const json& data, CharArrayFn callback = nullptr, bool callbackWithJSON=true) override {
        json result;
        result["content"] = CONTENT;
        return result.dump();
    }

    std::string slot_json(const json& data) override {
        json result;
        result["filename"] = SAVE_PATH;
        return result.dump();
    }

    void cancel_json(const json& data) override {
        cancelled_slot = data.at("id_slot");
    }

    std::string lora_weight_json(const json& data) override {
        json result;
        result["success"] = true;
        return result.dump();
    }

    std::string lora_list_json() override {
        json j = json::array();
        for (auto& l : LORAS)
            j.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});
        return j.dump();
    }

    std::string get_template_json()
    {
        return chat_template;
    }

    void set_template_json(const json& data)
    {
        chat_template = data.at("chat_template");
    }

    std::string apply_template_json(const json& data)
    {
        json result;
        result["prompt"] = data.at("messages")[0].at("message");
        return result.dump();
    }
};

void run_mock_tests() {
    TestLLM llm;

    // --- Tokenize ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"tokens", llm.TOKENS}};

        ASSERT(llm.build_tokenize_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_tokenize_json(output_json) == llm.TOKENS);
        ASSERT(llm.tokenize(llm.CONTENT.c_str()) == llm.TOKENS);
        ASSERT(llm.tokenize(llm.CONTENT) == llm.TOKENS);
        ASSERT(llm.tokenize_json(input_json) == output_json.dump());
    }

    // --- Detokenize ---
    {
        json input_json = {{"tokens", llm.TOKENS}};
        json output_json = {{"content", llm.CONTENT}};

        ASSERT(llm.build_detokenize_json(llm.TOKENS)["tokens"] == llm.TOKENS);
        ASSERT(llm.parse_detokenize_json(output_json) == llm.CONTENT);
        ASSERT(llm.detokenize(llm.TOKENS) == llm.CONTENT);
        ASSERT(llm.detokenize_json(input_json) == output_json.dump());
    }

    // --- Embeddings ---
    {
        json input_json = {{"content", llm.CONTENT}};
        json output_json = {{"embedding", llm.EMBEDDING}};

        ASSERT(llm.build_embeddings_json(llm.CONTENT) == input_json);
        ASSERT(llm.parse_embeddings_json(output_json) == llm.EMBEDDING);
        ASSERT(json::parse(llm.embeddings_json(input_json))["embedding"] == llm.EMBEDDING);
        ASSERT(llm.embeddings(llm.CONTENT) == llm.EMBEDDING);
        ASSERT(llm.embeddings(llm.CONTENT.c_str()) == llm.EMBEDDING);
    }

    // --- completion ---
    {
        int id_slot = 1;
        json params = {{"temperature", 0.7}};
        json input_json = {
            {"prompt", llm.CONTENT},
            {"id_slot", id_slot},
            {"n_keep", 0},
            {"temperature", 0.7}
        };
        json output_json = {{"content", llm.CONTENT}};

        llm.set_completion_params(params);
        ASSERT(llm.build_completion_json(llm.CONTENT, id_slot) == input_json);
        ASSERT(llm.parse_completion_json(output_json) == llm.CONTENT);
        ASSERT(llm.completion(llm.CONTENT, nullptr, id_slot) == llm.CONTENT);
        ASSERT(llm.completion_json(input_json) == output_json.dump());
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

        ASSERT(llm.build_slot_json(id_slot, action, llm.SAVE_PATH) == input_json);
        ASSERT(llm.parse_slot_json(output_json) == llm.SAVE_PATH);
        ASSERT(llm.slot(id_slot, action, llm.SAVE_PATH) == llm.SAVE_PATH);
        ASSERT(llm.slot_json(input_json) == output_json.dump());
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

        ASSERT(llm.build_lora_weight_json(loras) == input_json);
        ASSERT(llm.parse_lora_weight_json(output_json));
        ASSERT(llm.lora_weight_json(input_json) == output_json.dump());
        ASSERT(llm.lora_weight(loras));
    }

    // --- LoRA List ---
    {
        json input_json = json::array();
        for (const auto& l : llm.LORAS)
            input_json.push_back({{"id", l.id}, {"scale", l.scale}, {"path", l.path}});

        ASSERT(llm.parse_lora_list_json(input_json) == llm.LORAS);
        ASSERT(llm.lora_list() == llm.LORAS);
    }

    // --- Cancel ---
    {
        int id_slot = 10;
        json input_json = {{"id_slot", id_slot}};

        ASSERT(llm.build_cancel_json(id_slot) == input_json);
        llm.cancel(id_slot);
        ASSERT(llm.cancelled_slot == id_slot);
    }

    // --- Get / Set Template ---
    {
        std::string chat_template = "strange";
        json input_json = {{"chat_template", chat_template}};

        ASSERT(llm.build_set_template_json(chat_template) == input_json);
        llm.set_template_json(input_json);

        ASSERT(llm.parse_get_template_json(input_json) == chat_template);
        ASSERT(llm.get_template_json() == chat_template);
    }

    // --- Apply Template ---
    {
        std::string message = "hi";
        json messages_json = json::array();
        messages_json.push_back({{"message", message}});
        json input_json;
        input_json["messages"] = messages_json;
        json output_json;
        output_json["prompt"] = message;

        ASSERT(llm.build_apply_template_json(messages_json) == input_json);
        ASSERT(llm.parse_apply_template_json(output_json) == message);
        ASSERT(llm.apply_template_json(input_json) == output_json.dump());
    }
}

void run_LLM_tests(LLM* llm)
{
    llm->set_completion_params({{"seed", 0}, {"n_predict", 30}});

    for (bool use_api: {true, false}) {
        std::cout<<"*** USE_C_API: "<<use_api<<" ***"<<std::endl;
        test_tokenize(llm, use_api);
        test_completion(llm, false, use_api);
        test_completion(llm, true, use_api);
        test_embeddings(llm, use_api);
        test_apply_template(llm, use_api);
    }
}

void run_LLMAgent_tests(LLMLocal* llm) {
    test_ChatMessage();

    std::string system_prompt = "You are a helpful AI assistant for testing purposes.";
    LLMAgent* agent = new LLMAgent(llm, system_prompt);

    std::cout << std::endl << "<<< LLM agent" << std::endl;
    run_LLM_tests(agent);

    for (bool use_api: {true, false}) {
        std::cout<<"*** USE_C_API: "<<use_api<<" ***"<<std::endl;
        test_agent_chat(agent, false, use_api);
        test_agent_chat(agent, true, use_api);
        test_history(agent, use_api);
        test_save_history(agent, use_api);
    }

    if (LLMClient* client = dynamic_cast<LLMClient*>(llm)) {
        if (client->is_remote()) ASSERT(agent->get_slot() == -1);
        else ASSERT(agent->get_slot() > -1);
    } else ASSERT(agent->get_slot() > -1);

    delete agent;
    std::cout << ">>> LLM agent" << std::endl;
}

void run_LLMLocal_tests(LLMLocal* llm)
{
    run_LLM_tests(llm);
    run_LLMAgent_tests(llm);

    for (bool use_api: {true, false}) {
        std::cout<<"*** USE_C_API: "<<use_api<<" ***"<<std::endl;
        test_cancel(llm, use_api);
        test_slot(llm, use_api);
    }
}

void run_LLMProvider_tests(LLMProvider* llm)
{
    run_LLMLocal_tests(llm);

    for (bool use_api: {true, false}) {
        std::cout<<"*** USE_C_API: "<<use_api<<" ***"<<std::endl;
        test_lora_list(llm, use_api);
        test_template(llm, use_api);
    }
}

void set_SSL(LLMProvider* llm, LLMClient* llm_remote_client)
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

    LLM_Set_SSL(llm, server_crt.c_str(), server_key.c_str());
    llm_remote_client->set_SSL(client_crt.c_str());
}

int main(int argc, char** argv) {
    LLM_Debug(1);
    run_mock_tests();

    std::string model = "../tests/model.gguf";
    std::string command = "-m " + model;

#ifdef RUNTIME_TESTS
    std::cout << "-------- LLM runtime --------" << std::endl;
    LLMRuntime* llm_service = LLMRuntime::from_command(command);
#else
    std::cout << "-------- LLM service --------" << std::endl;
    LLMService* llm_service = new LLMService(model);
#endif
    LLM_Start(llm_service);

    EMBEDDING_SIZE = LLM_Embedding_Size(llm_service);
    run_LLMProvider_tests(llm_service);

    std::cout << std::endl << "-------- LLM client --------" << std::endl;
    LLMClient llm_client(llm_service);
    run_LLMLocal_tests(&llm_client);

    std::cout << std::endl << "-------- LLM remote client --------" << std::endl;
    LLMClient llm_remote_client("http://localhost", 8080);
    LLM_Start_Server(llm_service, "", 8080);
    run_LLMLocal_tests(&llm_remote_client);
    LLM_Stop_Server(llm_service);

    std::cout << std::endl << "-------- LLM remote client SSL --------" << std::endl;
    LLMClient llm_remote_client_SSL("https://localhost", 8080);
    set_SSL(llm_service, &llm_remote_client_SSL);
    LLM_Start_Server(llm_service, "", 8080);
    run_LLMLocal_tests(&llm_remote_client_SSL);

    std::cout << std::endl << "-------- Stop service --------" << std::endl;
    LLM_Delete(llm_service);
    return 0;
}
