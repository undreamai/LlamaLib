
#pragma once
#include <string>
#include <vector>
#include <functional>
#include "defs.h"
#include <iostream>

/**
 * @brief Handles concatenation of LLM response chunks (both streaming and non-streaming)
 * Accumulates content and tokens from multiple response chunks into a single result
 */
class ResponseConcatenator {
public:
    ResponseConcatenator() = default;

    /**
     * @brief Process a single chunk and accumulate its content/tokens
     * @param chunk_data The JSON chunk data (can be SSE format "data: {...}" or plain JSON)
     * @return true if processing should continue, false if done or error
     */
    bool process_chunk(const std::string& chunk_data);

    /**
     * @brief Build the final concatenated JSON result
     */
    json build_concatenated_json() const;

    /**
     * @brief Get the concatenated content string
     */
    const std::string& get_content() const { return concatenated_content_; }

    /**
     * @brief Get the concatenated tokens
     */
    const std::vector<int>& get_tokens() const { return concatenated_tokens_; }

    /**
     * @brief Get the complete result as JSON string
     */
    std::string get_result_json() const;

    /**
     * @brief Check if processing encountered an error
     */
    bool has_error() const { return has_error_; }

    /**
     * @brief Get the error JSON if any
     */
    const json& get_error() const { return error_; }

    /**
     * @brief Check if response is complete
     */
    bool is_complete() const { return is_complete_; }

    /**
     * @brief Get the number of chunks processed
     */
    size_t chunk_count() const { return chunk_count_; }

    /**
     * @brief Set a callback to be invoked after each chunk is processed
     */
    void set_callback(CharArrayFn callback, bool callWithJSON=false) {
        callback_ = std::move(callback);
        callWithJSON_ = callWithJSON;
    }

    /**
     * @brief Reset the concatenator state
     */
    void reset();

private:
    void accumulate_result(const json& item);

    std::string concatenated_content_;
    std::vector<int> concatenated_tokens_;
    json last_chunk_ = json::object();
    json error_ = json::object();
    bool has_error_ = false;
    bool is_complete_ = false;
    size_t chunk_count_ = 0;
    CharArrayFn callback_;
    bool callWithJSON_ = false;
};
