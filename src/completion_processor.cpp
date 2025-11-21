#include "completion_processor.h"

bool ResponseConcatenator::process_chunk(const std::string& chunk_data) {
    if (chunk_data.empty()) {
        return true;
    }

    std::string cleaned_chunk = chunk_data;

    // Handle SSE format
    // Remove "data: " prefix if present
    const std::string prefix = "data: ";
    if (cleaned_chunk.rfind(prefix, 0) == 0) {
        cleaned_chunk.erase(0, prefix.length());
    }

    // Remove trailing newlines/carriage returns
    while (!cleaned_chunk.empty() && 
            (cleaned_chunk.back() == '\n' || cleaned_chunk.back() == '\r')) {
        cleaned_chunk.pop_back();
    }

    // Check for SSE termination signal
    if (cleaned_chunk == "[DONE]") {
        is_complete_ = true;
        return false;
    }

    if (cleaned_chunk.empty()) {
        return true;
    }

    try {
        json chunk_json = json::parse(cleaned_chunk);

        // Handle error responses
        if (chunk_json.contains("error")) {
            error_ = chunk_json["error"];
            has_error_ = true;
            return false;
        }

        // Accumulate content
        if (chunk_json.contains("content")) {
            std::string content = chunk_json["content"].get<std::string>();
            concatenated_content_ += content;
        }

        // Accumulate tokens
        if (chunk_json.contains("tokens")) {
            for (const auto& tok : chunk_json["tokens"]) {
                concatenated_tokens_.push_back(tok.get<int>());
            }
        }

        // Store the last chunk for metadata (model, id, etc.)
        last_chunk_ = chunk_json;

        // Invoke callback if set
        if (callback_) {
            if (callWithJSON_) callback_(build_concatenated_json().dump().c_str());
            else callback_(concatenated_content_.c_str());
        }

        chunk_count_++;

    } catch (const json::exception& e) {
        error_ = json{{"message", std::string("JSON parse error: ") + e.what()}};
        has_error_ = true;
        return false;
    }

    return true;
}

json ResponseConcatenator::build_concatenated_json() const {
    json result = last_chunk_.empty() ? json::object() : last_chunk_;
    result["content"] = concatenated_content_;
    result["tokens"] = concatenated_tokens_;
    return result;
}

std::string ResponseConcatenator::get_result_json() const {
    if (has_error_) {
        return json{{"error", error_}}.dump();
    }
    return build_concatenated_json().dump();
}

void ResponseConcatenator::reset() {
    concatenated_content_.clear();
    concatenated_tokens_.clear();
    last_chunk_ = json::object();
    error_ = json::object();
    has_error_ = false;
    is_complete_ = false;
    chunk_count_ = 0;
}

void ResponseConcatenator::accumulate_result(const json& item) {
    if (item.contains("content")) {
        concatenated_content_ += item["content"].get<std::string>();
    }
    if (item.contains("tokens")) {
        for (const auto& tok : item["tokens"]) {
            concatenated_tokens_.push_back(tok.get<int>());
        }
    }
    last_chunk_ = item;
}
