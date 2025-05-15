#include "LLMFunctions.h"

//=========================== Tokenize ===========================//

json LLMFunctions::build_tokenize_json(const std::string& query)
{
    json j;
    j["content"] = query;
    return j;
}

std::vector<int> LLMFunctions::parse_tokenize_json(const json& result) {
    try {
        return result["tokens"].get<std::vector<int>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::vector<int> LLMFunctions::handle_tokenize(const json& data)
{
    return parse_tokenize_json(handle_tokenize_json(data));
}

std::string LLMFunctions::handle_tokenize_json(const std::string& input) {
    return handle_tokenize_json(build_tokenize_json(input));
}

std::vector<int> LLMFunctions::handle_tokenize(const std::string& input) {
    return handle_tokenize(build_tokenize_json(input));
}

std::string LLMFunctions::handle_tokenize_json(const char* input) {
    return handle_tokenize_json(std::string(input));
}

std::vector<int> LLMFunctions::handle_tokenize(const char* input) {
    return handle_tokenize(std::string(input));
}

//=========================== Detokenize ===========================//

json LLMFunctions::build_detokenize_json(const std::vector<int32_t>& tokens)
{
    json j;
    j["tokens"] = tokens;
    return j;
}

std::string LLMFunctions::parse_detokenize_json(const json& result) {
    try {
        return result["content"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLMFunctions::handle_detokenize(const json& data)
{
    return parse_detokenize_json(handle_detokenize_json(data));
}

std::string LLMFunctions::handle_detokenize_json(const std::vector<int32_t>& tokens) {
    return handle_detokenize_json(build_detokenize_json(tokens));
}

std::string LLMFunctions::handle_detokenize(const std::vector<int32_t>& tokens) {
    return handle_detokenize(build_detokenize_json(tokens));
}

//=========================== Embeddings ===========================//

json LLMFunctions::build_embeddings_json(const std::string& query)
{
    json j;
    j["content"] = query;
    return j;
}

std::vector<float> LLMFunctions::parse_embeddings_json(const json& result) {
    try {
        return result["embedding"].get<std::vector<float>>();
    }
    catch (const std::exception&) {}
    return {};
}

std::vector<float> LLMFunctions::handle_embeddings(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return parse_embeddings_json(handle_embeddings_json(data));
}

std::string LLMFunctions::handle_embeddings_json(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings_json(build_embeddings_json(query), res, is_connection_closed);
}

std::vector<float> LLMFunctions::handle_embeddings(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings(build_embeddings_json(query), res, is_connection_closed);
}

std::string LLMFunctions::handle_embeddings_json(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings_json(std::string(query), res, is_connection_closed);
}

std::vector<float> LLMFunctions::handle_embeddings(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings(std::string(query), res, is_connection_closed);
}

//=========================== Completion ===========================//

json LLMFunctions::build_completions_json(const std::string& prompt, int id_slot, const json& params)
{
    json j;
    j["prompt"] = prompt;
    j["id_slot"] = id_slot;
    for (json::const_iterator it = params.begin(); it != params.end(); ++it) {
        j[it.key()] = it.value();
    }
    return j;
}

std::string LLMFunctions::parse_completions_json(const json& result)
{
    try {
        return result["content"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLMFunctions::handle_completions(const json& data, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return parse_completions_json(handle_completions_json(data, stringWrapper, res, is_connection_closed, oaicompat));
}

std::string LLMFunctions::handle_completions_json(const std::string& prompt, int id_slot, const json& params, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return handle_completions_json(build_completions_json(prompt, params, id_slot), stringWrapper, res, is_connection_closed, oaicompat);
}

std::string LLMFunctions::handle_completions(const std::string& prompt, int id_slot, const json& params, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return handle_completions(build_completions_json(prompt, params, id_slot), stringWrapper, res, is_connection_closed, oaicompat);
}

//=========================== Lora Adapters Apply ===========================//

json LLMFunctions::build_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras)
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

bool LLMFunctions::parse_lora_adapters_apply_json(const json& result) {
    try {
        return result["success"].get<bool>();
    }
    catch (const std::exception&) {}
    return false;
}

bool LLMFunctions::handle_lora_adapters_apply(const json& data, httplib::Response* res)
{
    return parse_lora_adapters_apply_json(handle_lora_adapters_apply_json(data, res));
}

std::string LLMFunctions::handle_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return handle_lora_adapters_apply_json(build_lora_adapters_apply_json(loras), res);
}

bool LLMFunctions::handle_lora_adapters_apply(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return handle_lora_adapters_apply(build_lora_adapters_apply_json(loras), res);
}

//=========================== Lora Adapters List ===========================//

std::vector<LoraIdScalePath> LLMFunctions::parse_lora_adapters_list_json(const json& result)
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

std::vector<LoraIdScalePath> LLMFunctions::handle_lora_adapters_list()
{
    return parse_lora_adapters_list_json(json::parse(handle_lora_adapters_list_json()));
}

//=========================== Slot Action ===========================//

json LLMFunctions::build_slots_action_json(int id_slot, std::string action, std::string filepath)
{
    json j;
    j["id_slot"] = id_slot;
    j["action"] = action;
    j["filepath"] = filepath;
    return j;
}

std::string LLMFunctions::parse_slots_action_json(const json& result)
{
    try {
        return result["filename"].get<std::string>();
    }
    catch (const std::exception&) {}
    return false;
}

std::string LLMFunctions::handle_slots_action_json(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return handle_slots_action_json(build_slots_action_json(id_slot, action, filepath), res);
}

std::string LLMFunctions::handle_slots_action(const json& data, httplib::Response* res)
{
    return parse_slots_action_json(handle_slots_action_json(data, res));
}

std::string LLMFunctions::handle_slots_action(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return handle_slots_action(build_slots_action_json(id_slot, action, filepath), res);
}