#pragma once
#include <string>
#include "json.hpp"

using json = nlohmann::ordered_json;

// DEFINE_INLINE_JSON_OVERLOADS(tokenize, const std::string&, std::vector<int>)
//   will be translated to implementation of:
// virtual std::string handle_tokenize_json(const std::string& query);
// virtual std::vector<int> handle_tokenize(const std::string& query);

#define DEFINE_INLINE_JSON_OVERLOADS(FUNC_NAME, INPUT_TYPE, RETURN_TYPE)           \
    inline std::string handle_##FUNC_NAME##_json(INPUT_TYPE input) {               \
        return handle_##FUNC_NAME##_json(build_##FUNC_NAME##_json(input));         \
    }                                                                              \
    inline RETURN_TYPE handle_##FUNC_NAME(INPUT_TYPE input) {                      \
        return handle_##FUNC_NAME(build_##FUNC_NAME##_json(input));                \
    }

// DEFINE_INLINE_CHAR_JSON_OVERLOADS(tokenize, std::vector<int>)
//   will be translated to implementation of:
// virtual std::string handle_tokenize_json(const char* query);
// virtual std::vector<int> handle_tokenize(const char* query);

#define DEFINE_INLINE_CHAR_JSON_OVERLOADS(FUNC_NAME, RETURN_TYPE)                  \
    inline std::string handle_##FUNC_NAME##_json(const char* input) {              \
        return handle_##FUNC_NAME##_json(std::string(input));                      \
    }                                                                              \
    inline RETURN_TYPE handle_##FUNC_NAME(const char* input) {                     \
        return handle_##FUNC_NAME(std::string(input));                             \
    }


class LLMFunctions {
public:
    virtual ~LLMFunctions() = default;

    virtual std::string handle_tokenize_json(const json data) = 0;
    virtual json build_tokenize_json(const std::string& query);
    virtual std::vector<int> handle_tokenize(const json data);
    DEFINE_INLINE_JSON_OVERLOADS(tokenize, const std::string&, std::vector<int>)
    DEFINE_INLINE_CHAR_JSON_OVERLOADS(tokenize, std::vector<int>)
        
    virtual std::string handle_detokenize_json(const json data) = 0;
    virtual json build_detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string handle_detokenize(const json data);
    DEFINE_INLINE_JSON_OVERLOADS(detokenize, const std::vector<int32_t>&, std::string)
};


/* TODO
std::string handle_embeddings(json data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_true);
std::string handle_lora_adapters_apply(json data, httplib::Response* res = nullptr);
std::string handle_lora_adapters_list();
std::string handle_completions(json data, StringWrapper* stringWrapper = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_true, oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE);
std::string handle_slots_action(json data, httplib::Response* res = nullptr);
void handle_cancel_action(int id_slot);*/