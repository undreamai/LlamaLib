#include "LLM_client.h"

LLMClient::LLMClient(LLMProvider* llm_): llm(llm_){ }

LLMClient::LLMClient(LLMLib* llmLib) : LLMClient((LLMProvider*) llmLib->llm) { }

std::string LLMClient::handle_tokenize_impl(const json& data)
{
    return llm->handle_tokenize_impl(data);
}

std::string LLMClient::handle_detokenize_impl(const json& data)
{
    return llm->handle_detokenize_impl(data);
}

std::string LLMClient::handle_embeddings_impl(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return llm->handle_embeddings_impl(data, res, is_connection_closed);
}

std::string LLMClient::handle_completions_impl(const json& data, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return llm->handle_completions_impl(data, stringWrapper, res, is_connection_closed, oaicompat);
}

std::string LLMClient::handle_slots_action_impl(const json& data, httplib::Response* res)
{
    return llm->handle_slots_action_impl(data, res);
}

void LLMClient::handle_cancel_action_impl(int id_slot)
{
    llm->handle_cancel_action_impl(id_slot);
}


//================ Remote requests ================//

RemoteLLMClient::RemoteLLMClient(const std::string& url_, const int port_) : url(url_), port(port_) { }

static size_t StreamingWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t totalSize = size * nmemb;
    StreamingContext* ctx = static_cast<StreamingContext*>(userp);

    if (ctx && contents) {
        std::string chunk(static_cast<char*>(contents), totalSize);
        ctx->buffer += chunk;
        if (ctx->stringWrapper != nullptr){
            ctx->stringWrapper->SetContent(ctx->buffer);
        }
    }
    
    return totalSize;
}

std::string RemoteLLMClient::post_request(
    const std::string& url, 
    int port, 
    const std::string& path, 
    const json& payload,
    StringWrapper* stringWrapper
) {
    CURL* curl = curl_easy_init();
    
    if (!curl) {
        std::cerr << "CURL initialization failed" << std::endl;
        return "";
    }
        
    StreamingContext context;
    context.stringWrapper = stringWrapper;
    
    try {
        std::ostringstream full_url;
        full_url << url << ":" << port << "/" << path;

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Accept: text/event-stream");
        headers = curl_slist_append(headers, "Cache-Control: no-cache");

        std::string payload_str = payload.dump();
        
        curl_easy_setopt(curl, CURLOPT_URL, full_url.str().c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_str.length());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamingWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK && res != CURLE_ABORTED_BY_CALLBACK) {
            std::string error = "CURL streaming failed: " + std::string(curl_easy_strerror(res));
            std::cerr << error << std::endl;
        }

        curl_slist_free_all(headers);
        
    } catch (const std::exception& e) {
        std::string error = "Exception in streaming request: " + std::string(e.what());
        context.buffer = error;
        std::cerr << error << std::endl;
    }
    
    curl_easy_cleanup(curl);
    return context.buffer;
}


std::string RemoteLLMClient::handle_tokenize_impl(const json& data)
{
    return post_request(url, port, "tokenize", data);
}

std::string RemoteLLMClient::handle_detokenize_impl(const json& data)
{
    return post_request(url, port, "detokenize", data);
}

std::string RemoteLLMClient::handle_embeddings_impl(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return post_request(url, port, "embeddings", data);
}

std::string RemoteLLMClient::handle_completions_impl(const json& data, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return post_request(url, port, "completion", data, stringWrapper);
}
