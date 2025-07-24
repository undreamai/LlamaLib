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
    if (!LLMProviderRegistry::initialised)
    {
        static std::once_flag initialized;
        std::call_once(initialized, []() {
            set_error_handlers();
            register_sigint_hook(llm_sigint_signal_handler);
        });
    }
}

LLMProviderRegistry* LLMProviderRegistry::custom_instance_ = nullptr;
bool LLMProviderRegistry::initialised = false;

//=========================== Helpers ===========================//

std::string LLM::LLM_args_to_command(const std::string& model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, const std::vector<std::string>& lora_paths)
{
    std::string command = "-m " + model_path +
                          " -t " + std::to_string(num_threads) +
                          " -np " + std::to_string(num_parallel) +
                          " -c " + std::to_string(context_size) +
                          " -b " + std::to_string(batch_size);

    if (num_GPU_layers > 0) command += " -ngl " + std::to_string(num_GPU_layers);
    if (flash_attention) command += " --flash-attn";
    if (embedding_only) command += " --embedding";
    for (const auto& lora_path : lora_paths) command += " --lora " + lora_path;
    return command;
}

bool LLM::has_gpu_layers(const std::string& command) {
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

//=========================== Get Template ===========================//

std::string LLM::parse_get_template_json(const json& result) {
    try {
        return result.at("chat_template").get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::get_template() {
    return parse_get_template_json(json::parse(get_template_json()));
}

//=========================== Apply Template ===========================//

json LLM::build_apply_template_json(const json& messages)
{
    json j;
    j["messages"] = messages;
    return j;
}

std::string LLM::parse_apply_template_json(const json& result) {
    try {
        return result.at("prompt").get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::apply_template(const json& messages) {
    return parse_apply_template_json(json::parse(apply_template_json(build_apply_template_json(messages))));
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
        return result.at("tokens").get<std::vector<int>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::vector<int> LLM::tokenize(const std::string& input) {
    return parse_tokenize_json(json::parse(tokenize_json(build_tokenize_json(input))));
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
        return result.at("content").get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::detokenize(const std::vector<int32_t>& tokens) {
    return parse_detokenize_json(json::parse(detokenize_json(build_detokenize_json(tokens))));
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
        return result.at("embedding").get<std::vector<float>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::vector<float> LLM::embeddings(const std::string& query)
{
    return parse_embeddings_json(json::parse(embeddings_json(build_embeddings_json(query))));
}

//=========================== Completion ===========================//

json LLM::build_completion_json(const std::string& prompt, int id_slot, const json& params)
{
    json j;
    j["prompt"] = prompt;
    j["id_slot"] = id_slot;
    j["seed"] = seed;
    j["n_predict"] = n_predict;
    j["n_keep"] = n_keep;
    j["temperature"] = temperature;
    if (json_schema != "") j["json_schema"] = json_schema;
    else if (grammar != "") j["grammar"] = grammar;

    if (params.is_object()) {
        for (json::const_iterator it = params.begin(); it != params.end(); ++it) {
            j[it.key()] = it.value();
        }
    }
    return j;
}

std::string LLM::parse_completion_json(const json& result)
{
    try {
        return result.at("content").get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::completion(const std::string& prompt, CharArrayFn callback, int id_slot, const json& params)
{
    return parse_completion_json(json::parse(completion_json(
        build_completion_json(prompt, id_slot, params),
        callback,
        false
    )));
}

//=========================== Cancel ===========================//

json LLMLocal::build_cancel_json(int id_slot)
{
    json j;
    j["id_slot"] = id_slot;
    return j;
}

void LLMLocal::cancel(int id_slot) {
    cancel_json(build_cancel_json(id_slot));
}

//=========================== Slot Action ===========================//

json LLMLocal::build_slot_json(int id_slot, const std::string& action, const std::string& filepath)
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
        return result.at("filename").get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLMLocal::slot(int id_slot, const std::string& action, const std::string& filepath)
{
    return parse_slot_json(json::parse(slot_json(build_slot_json(id_slot, action, filepath))));
}

//=========================== Logging ===========================//

void LLMProvider::logging_stop()
{
    logging_callback(nullptr);
}

//=========================== Set Template ===========================//

json LLMProvider::build_set_template_json(std::string chat_template)
{
    json j;
    j["chat_template"] = chat_template;
    return j;
}

void LLMProvider::set_template(std::string chat_template) {
    set_template_json(build_set_template_json(chat_template));
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
        return result.at("success").get<bool>();
    }
    catch (const std::exception&) {}
    return false;
}

bool LLMProvider::lora_weight(const std::vector<LoraIdScale>& loras)
{
    return parse_lora_weight_json(json::parse(lora_weight_json(build_lora_weight_json(loras))));
}

//=========================== Lora Adapters List ===========================//

std::string LLMProvider::lora_list_json()
{
    return lora_list_json();
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

bool Has_GPU_Layers(const char* command)
{
    return LLM::has_gpu_layers(command);
}

void LLM_Debug(int debug_level)
{
    LLMProviderRegistry& registry = LLMProviderRegistry::instance();
    registry.set_debug_level(debug_level);
    for (auto* inst : registry.get_instances()) {
        inst->debug(debug_level);
    }
}

void LLM_Logging_Callback(CharArrayFn callback)
{
    LLMProviderRegistry& registry = LLMProviderRegistry::instance();
    registry.set_log_callback(callback);
    for (auto* inst : registry.get_instances()) {
        inst->logging_callback(callback);
    }
}

void LLM_Logging_Stop()
{
    LLM_Logging_Callback(nullptr);
}

#ifdef _DEBUG
const bool IsDebuggerAttached(void) {
#ifdef _MSC_VER
    return ::IsDebuggerPresent();
#elif __APPLE__
    return AmIBeingDebugged();
#elif __linux__
    return debuggerIsAttached();
#else
    return false;
#endif
}
#endif

const char* LLM_Tokenize(LLM* llm, const char* query) {
    json result = llm->tokenize(query);
    return stringToCharArray(result.dump());
}

const char* LLM_Detokenize(LLM* llm, const char* tokens_as_json) {
    return stringToCharArray(llm->detokenize(json::parse(tokens_as_json)));
}

const char* LLM_Embeddings(LLM* llm, const char* query) {
    json result = llm->embeddings(query);
    return stringToCharArray(result.dump());
}

const char* LLM_Completion(LLM* llm, const char* prompt, CharArrayFn callback, int id_slot, const char* params_json) {
    json params = json::parse(params_json ? params_json : "{}");
    return stringToCharArray(llm->completion(prompt, callback, id_slot, params));
}

const char* LLM_Completion_JSON(LLM* llm, const char* prompt, CharArrayFn callback, int id_slot, const char* params_json) {
    json params = json::parse(params_json ? params_json : "{}");
    std::string completion_json_str = llm->completion_json(
        llm->build_completion_json(std::string(prompt), id_slot, params),
        callback,
        true
    );
    return stringToCharArray(completion_json_str);
}

const char* LLM_Get_Template(LLM* llm) {
    return stringToCharArray(llm->get_template());
}

const char* LLM_Apply_Template(LLM* llm, const char* messages_as_json) {
    return stringToCharArray(llm->apply_template(json::parse(messages_as_json)));
}

void LLM_Set_Template(LLMProvider* llm, const char* chat_template) {
    llm->set_template(chat_template);
}

const char* LLM_Slot(LLMLocal* llm, int id_slot, const char* action, const char* filepath) {
    return stringToCharArray(llm->slot(id_slot, action, filepath));
}

void LLM_Cancel(LLMLocal* llm, int id_slot) {
    llm->cancel(id_slot);
}

bool LLM_Lora_Weight(LLMProvider* llm, const char* loras_as_json) {
    return llm->parse_lora_weight_json(json::parse(llm->lora_weight_json(loras_as_json)));
}

const char* LLM_Lora_List(LLMProvider* llm) {
    std::string result = llm->lora_list_json();
    return stringToCharArray(result);
}

void LLM_Delete(LLMProvider* llm) {
    if (llm != nullptr)
    {
        delete llm;
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

const int LLM_Status_Code() {
    return get_status_code();
}

const char* LLM_Status_Message() {
    std::string result = get_status_message();
    return stringToCharArray(result);
}

const int LLM_Embedding_Size(LLMProvider* llm) {
    return llm->embedding_size();
}
