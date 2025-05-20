#include "LLM.h"

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

std::vector<int> LLM::handle_tokenize(const json& data)
{
    return parse_tokenize_json(handle_tokenize_json(data));
}

std::string LLM::handle_tokenize_json(const std::string& input) {
    return handle_tokenize_json(build_tokenize_json(input));
}

std::vector<int> LLM::handle_tokenize(const std::string& input) {
    return handle_tokenize(build_tokenize_json(input));
}

std::string LLM::handle_tokenize_json(const char* input) {
    return handle_tokenize_json(std::string(input));
}

std::vector<int> LLM::handle_tokenize(const char* input) {
    return handle_tokenize(std::string(input));
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

std::string LLM::handle_detokenize(const json& data)
{
    return parse_detokenize_json(handle_detokenize_json(data));
}

std::string LLM::handle_detokenize_json(const std::vector<int32_t>& tokens) {
    return handle_detokenize_json(build_detokenize_json(tokens));
}

std::string LLM::handle_detokenize(const std::vector<int32_t>& tokens) {
    return handle_detokenize(build_detokenize_json(tokens));
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

std::vector<float> LLM::handle_embeddings(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return parse_embeddings_json(handle_embeddings_json(data));
}

std::string LLM::handle_embeddings_json(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings_json(build_embeddings_json(query), res, is_connection_closed);
}

std::vector<float> LLM::handle_embeddings(const std::string& query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings(build_embeddings_json(query), res, is_connection_closed);
}

std::string LLM::handle_embeddings_json(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings_json(std::string(query), res, is_connection_closed);
}

std::vector<float> LLM::handle_embeddings(const char* query, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return handle_embeddings(std::string(query), res, is_connection_closed);
}

//=========================== Completion ===========================//

json LLM::build_completions_json(const std::string& prompt, int id_slot, const json& params)
{
    json j;
    j["prompt"] = prompt;
    j["id_slot"] = id_slot;
    for (json::const_iterator it = params.begin(); it != params.end(); ++it) {
        j[it.key()] = it.value();
    }
    return j;
}

std::string LLM::parse_completions_json(const json& result)
{
    try {
        return result["content"].get<std::string>();
    }
    catch (const std::exception&) {}
    return "";
}

std::string LLM::handle_completions(const json& data, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return parse_completions_json(handle_completions_json(data, stringWrapper, res, is_connection_closed, oaicompat));
}

std::string LLM::handle_completions_json(const std::string& prompt, int id_slot, const json& params, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return handle_completions_json(build_completions_json(prompt, params, id_slot), stringWrapper, res, is_connection_closed, oaicompat);
}

std::string LLM::handle_completions(const std::string& prompt, int id_slot, const json& params, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return handle_completions(build_completions_json(prompt, params, id_slot), stringWrapper, res, is_connection_closed, oaicompat);
}

//=========================== Lora Adapters Apply ===========================//

json LLM::build_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras)
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

bool LLM::parse_lora_adapters_apply_json(const json& result) {
    try {
        return result["success"].get<bool>();
    }
    catch (const std::exception&) {}
    return false;
}

bool LLM::handle_lora_adapters_apply(const json& data, httplib::Response* res)
{
    return parse_lora_adapters_apply_json(handle_lora_adapters_apply_json(data, res));
}

std::string LLM::handle_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return handle_lora_adapters_apply_json(build_lora_adapters_apply_json(loras), res);
}

bool LLM::handle_lora_adapters_apply(const std::vector<LoraIdScale>& loras, httplib::Response* res)
{
    return handle_lora_adapters_apply(build_lora_adapters_apply_json(loras), res);
}

//=========================== Lora Adapters List ===========================//

std::vector<LoraIdScalePath> LLM::parse_lora_adapters_list_json(const json& result)
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

std::vector<LoraIdScalePath> LLM::handle_lora_adapters_list()
{
    return parse_lora_adapters_list_json(json::parse(handle_lora_adapters_list_json()));
}

//=========================== Slot Action ===========================//

json LLM::build_slots_action_json(int id_slot, std::string action, std::string filepath)
{
    json j;
    j["id_slot"] = id_slot;
    j["action"] = action;
    j["filepath"] = filepath;
    return j;
}

std::string LLM::parse_slots_action_json(const json& result)
{
    try {
        return result["filename"].get<std::string>();
    }
    catch (const std::exception&) {}
    return false;
}

std::string LLM::handle_slots_action_json(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return handle_slots_action_json(build_slots_action_json(id_slot, action, filepath), res);
}

std::string LLM::handle_slots_action(const json& data, httplib::Response* res)
{
    return parse_slots_action_json(handle_slots_action_json(data, res));
}

std::string LLM::handle_slots_action(int id_slot, std::string action, std::string filepath, httplib::Response* res)
{
    return handle_slots_action(build_slots_action_json(id_slot, action, filepath), res);
}

//=========================== API ===========================//

const int LLM_Test() {
    return 100;
}

const void LLM_Tokenize(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    wrapper->SetContent(llm->handle_tokenize_json(json::parse(json_data)));
}

const void LLM_Detokenize(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    wrapper->SetContent(llm->handle_detokenize(json::parse(json_data)));
}

const void LLM_Embeddings(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_embeddings_json(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Lora_Weight(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_lora_adapters_apply_json(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Lora_List(LLM* llm, StringWrapper* wrapper) {
    std::string result = llm->handle_lora_adapters_list_json();
    wrapper->SetContent(result);
}

const void LLM_Completion(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_completions_json(json::parse(json_data), wrapper);
    wrapper->SetContent(result);
}

const void LLM_Slot(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_slots_action_json(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Cancel(LLM* llm, int id_slot) {
    llm->handle_cancel_action(id_slot);
}
