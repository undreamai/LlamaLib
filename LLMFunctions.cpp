#include "LLMFunctions.h"

//=========================== Tokenize ===========================//

json LLMFunctions::build_tokenize_json(const std::string& query) {
    json j;
    j["content"] = query;
    return j;
}

std::vector<int> LLMFunctions::handle_tokenize(const json data) {
    std::string json_str = handle_tokenize_json(data);
    try {
        auto j = nlohmann::json::parse(json_str);
        return j["tokens"].get<std::vector<int>>();
    }
    catch (...) {
        return {};
    }
}

//=========================== Detokenize ===========================//

json LLMFunctions::build_detokenize_json(const std::vector<int32_t>& tokens) {
    json j;
    j["tokens"] = tokens;
    return j;
}

std::string LLMFunctions::handle_detokenize(const json data) {
    std::string json_str = handle_detokenize_json(data);
    try {
        auto j = nlohmann::json::parse(json_str);
        return j["content"].get<std::string>();
    }
    catch (...) {
        return {};
    }
}