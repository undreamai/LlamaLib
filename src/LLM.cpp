#include "LLM.h"

std::atomic_flag sigint_terminating = ATOMIC_FLAG_INIT;

void llm_sigint_signal_handler(int sig) {
    if (sigint_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    for (auto* inst : LLMProviderRegistry::instance().get_instances()) {
        inst->stop();
        inst->stop_server();
    }
}

// Use a function to ensure the setup only happens once across all libraries
void ensure_error_handlers_initialized() {
    static std::once_flag initialized;
    std::call_once(initialized, []() {
        set_error_handlers();
        register_sigint_hook(llm_sigint_signal_handler);
    });
}

//=========================== Helpers ===========================//

std::string LLM::LLM_args_to_command(const char* model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, int lora_count, const char** lora_paths)
{
    std::string command = std::string("-m ") + model_path
                        + " -t " + std::to_string(num_threads)
                        + " -ngl " + std::to_string(num_GPU_layers)
                        + " -np " + std::to_string(num_parallel)
                        + " -c " + std::to_string(context_size)
                        + " -b " + std::to_string(batch_size);
    if (flash_attention) command += " --flash-attn";
    if (embedding_only) command += " --embedding";
    if (lora_paths != nullptr && lora_count > 0)
    {
        for (int i = 0; i < lora_count; ++i) {
            command += " --lora " + std::string(lora_paths[i]);
        }
    }
    return command;
}

bool has_gpu_layers(const std::string& command) {
    std::istringstream iss(command);
    std::vector<std::string> args;
    std::string token;

    // Simple splitting (does not handle quoted args)
    while (iss >> token) {
        args.push_back(token);
    }

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];

        // Match separate argument + value
        if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            if (i + 1 < args.size()) {
                try {
                    int val = std::stoi(args[i + 1]);
                    return val > 0;
                } catch (...) {
                    continue;
                }
            }
        }

        // Match inline --flag=value
        size_t eqPos = arg.find('=');
        if (eqPos != std::string::npos) {
            std::string key = arg.substr(0, eqPos);
            std::string value = arg.substr(eqPos + 1);

            if (key == "-ngl" || key == "--gpu-layers" || key == "--n-gpu-layers") {
                try {
                    int val = std::stoi(value);
                    return val > 0;
                } catch (...) {
                    continue;
                }
            }
        }
    }

    return false;
}

//=========================== Tokenize ===========================//

json LLM::build_tokenize_json(const std::string& query)
{
    json j;
    j["content"] = query;
    return j;
}

std::vector<int> LLM::parse_tokenize_json(const json& result) {
    try {
        return result["tokens"].get<std::vector<int>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::string LLM::tokenize_json(const json& input) {
    return tokenize_impl(input);
}

std::string LLM::tokenize_json(const std::string& input) {
    return tokenize_json(build_tokenize_json(input));
}

std::string LLM::tokenize_json(const char* input) {
    return tokenize_json(std::string(input));
}

std::vector<int> LLM::tokenize(const json& data)
{
    return parse_tokenize_json(json::parse(tokenize_json(data)));
}

std::vector<int> LLM::tokenize(const std::string& input) {
    return tokenize(build_tokenize_json(input));
}

std::vector<int> LLM::tokenize(const char* input) {
    return tokenize(std::string(input));
}

//=========================== Detokenize ===========================//

json LLM::build_detokenize_json(const std::vector<int32_t>& tokens)
{
    json j;
    j["tokens"] = tokens;
    return j;
}

std::string LLM::parse_detokenize_json(const json& result) {
    try {
        return result["content"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::detokenize_json(const json& input) {
    return detokenize_impl(input);
}

std::string LLM::detokenize_json(const std::vector<int32_t>& tokens) {
    return detokenize_json(build_detokenize_json(tokens));
}

std::string LLM::detokenize(const json& data)
{
    return parse_detokenize_json(json::parse(detokenize_json(data)));
}

std::string LLM::detokenize(const std::vector<int32_t>& tokens) {
    return detokenize(build_detokenize_json(tokens));
}

//=========================== Embeddings ===========================//

json LLM::build_embeddings_json(const std::string& query)
{
    json j;
    j["content"] = query;
    return j;
}

std::vector<float> LLM::parse_embeddings_json(const json& result) {
    try {
        return result["embedding"].get<std::vector<float>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::string LLM::embeddings_json(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return embeddings_impl(data, res, is_connection_closed);
}

std::string LLM::embeddings_json(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return embeddings_json(build_embeddings_json(query), res, is_connection_closed);
}

std::string LLM::embeddings_json(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return embeddings_json(std::string(query), res, is_connection_closed);
}

std::vector<float> LLM::embeddings(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return parse_embeddings_json(json::parse(embeddings_json(data)));
}

std::vector<float> LLM::embeddings(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return embeddings(build_embeddings_json(query), res, is_connection_closed);
}

std::vector<float> LLM::embeddings(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return embeddings(std::string(query), res, is_connection_closed);
}

//=========================== Completion ===========================//

json LLM::build_completion_json(const std::string& prompt, int id_slot, const json& params)
{
    json j;
    j["prompt"] = prompt;
    j["id_slot"] = id_slot;
    for (json::const_iterator it = params.begin(); it != params.end(); ++it) {
        j[it.key()] = it.value();
    }
    return j;
}

std::string LLM::parse_completion_json(const json& result)
{
    try {
        return result["content"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::completion_json(const json& data, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return completion_impl(data, callback, res, is_connection_closed, oaicompat);
}

std::string LLM::completion_json(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return completion_json(build_completion_json(prompt, id_slot, params), callback, res, is_connection_closed, oaicompat);
}

std::string LLM::completion(const json& data, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return parse_completion_json(json::parse(completion_json(data, callback, res, is_connection_closed, oaicompat)));
}

std::string LLM::completion(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return completion(build_completion_json(prompt, id_slot, params), callback, res, is_connection_closed, oaicompat);
}

//=========================== Slot Action ===========================//

json LLMLocal::build_slot_json(int id_slot, std::string action, std::string filepath)
{
    json j;
    j["id_slot"] = id_slot;
    j["action"] = action;
    j["filepath"] = filepath;
    return j;
}

std::string LLMLocal::parse_slot_json(const json& result)
{
    try {
        return result["filename"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLMLocal::slot_json(const json& data, httplib::Response* res)
{
    return slot_impl(data, res);
}

std::string LLMLocal::slot_json(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return slot_json(build_slot_json(id_slot, action, filepath), res);
}

std::string LLMLocal::slot(const json& data, httplib::Response* res)
{
    return parse_slot_json(json::parse(slot_json(data, res)));
}

std::string LLMLocal::slot(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return slot(build_slot_json(id_slot, action, filepath), res);
}


//=========================== Cancel ===========================//
    
void LLMLocal::cancel(int id_slot)
{
    cancel_impl(id_slot);
}


//=========================== Lora Adapters Apply ===========================//

json LLMProvider::build_lora_weight_json(const std::vector<LoraIdScale>& loras)
{
    json j = json::array();
    for (const auto& lora : loras) {
        j.push_back({
            {"id", lora.id},
            {"scale", lora.scale}
            });
    }
    return j;
}

bool LLMProvider::parse_lora_weight_json(const json& result) {
    try {
        return result["success"].get<bool>();
    }
    catch (const std::exception&) {}
    return false;
}

std::string LLMProvider::lora_weight_json(const json& data, httplib::Response* res)
{
    return lora_weight_impl(data, res);
}

std::string LLMProvider::lora_weight_json(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return lora_weight_json(build_lora_weight_json(loras), res);
}

bool LLMProvider::lora_weight(const json& data, httplib::Response* res)
{
    return parse_lora_weight_json(json::parse(lora_weight_json(data, res)));
}

bool LLMProvider::lora_weight(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return lora_weight(build_lora_weight_json(loras), res);
}

//=========================== Lora Adapters List ===========================//

std::string LLMProvider::lora_list_json()
{
    return lora_list_impl();
}

std::vector<LoraIdScalePath> LLMProvider::parse_lora_list_json(const json& result)
{
    std::vector<LoraIdScalePath> loras;
    try {
        for (const auto& lora : result) {
            loras.push_back({
                lora["id"].get<int>(),
                lora["scale"].get<float>(),
                lora["path"].get<std::string>()
                });
        }
    }
    catch (const std::exception&) {}
    return loras;
}

std::vector<LoraIdScalePath> LLMProvider::lora_list()
{
    return parse_lora_list_json(json::parse(lora_list_json()));
}

//=========================== API ===========================//

bool Has_GPU_Layers(const std::string& command)
{
    return has_gpu_layers(command);
}

const char* LLM_Tokenize(LLM* llm, const char* json_data) {
    std::string result = llm->tokenize_json(json::parse(json_data));
    return stringToCharArray(result);
}

const char* LLM_Detokenize(LLM* llm, const char* json_data) {
    std::string result = llm->detokenize_json(json::parse(json_data));
    return stringToCharArray(result);
}

const char* LLM_Embeddings(LLM* llm, const char* json_data) {
    std::string result = llm->embeddings_json(json::parse(json_data));
    return stringToCharArray(result);
}

const char* LLM_Completion(LLM* llm, const char* json_data, CharArrayFn callback) {
    std::string result = llm->completion_json(json::parse(json_data), callback);
    return stringToCharArray(result);
}

const char* LLM_Slot(LLMLocal* llm, const char* json_data) {
    std::string result = llm->slot_json(json::parse(json_data));
    return stringToCharArray(result);
}

void LLM_Cancel(LLMLocal* llm, int id_slot) {
    llm->cancel(id_slot);
}

const char* LLM_Lora_Weight(LLMProvider* llm, const char* json_data) {
    std::string result = llm->lora_weight_json(json::parse(json_data));
    return stringToCharArray(result);
}

const char* LLM_Lora_List(LLMProvider* llm) {
    std::string result = llm->lora_list_json();
    return stringToCharArray(result);
}

void LLM_Delete(LLMProvider* llm) {
    if (llm != nullptr)
    {
        LOG_INFO("Deleting LLM service", {});
        delete llm;
        llm = nullptr;
    }
}

void LLM_Start_Server(LLMProvider* llm, const char* host, int port, const char* API_key) {
    llm->start_server(host, port, API_key);
}

void LLM_Stop_Server(LLMProvider* llm) {
    llm->stop_server();
}

void LLM_Join_Service(LLMProvider* llm)
{
    llm->join_service();
}

void LLM_Join_Server(LLMProvider* llm)
{
    llm->join_server();
}

void LLM_Start(LLMProvider* llm) {
    llm->start();
}

const bool LLM_Started(LLMProvider* llm) {
    return llm->started();
}

void LLM_Stop(LLMProvider* llm) {
    llm->stop();
}

void LLM_Set_SSL(LLMProvider* llm, const char* SSL_cert, const char* SSL_key){
    llm->set_SSL(SSL_cert, SSL_key);
}

const int LLM_Status_Code(LLMProvider* llm) {
    return get_status_code();
}

const char* LLM_Status_Message(LLMProvider* llm) {
    std::string result = get_status_message();
    return stringToCharArray(result);
}

const int LLM_Embedding_Size(LLMProvider* llm) {
    return llm->embedding_size();
}
