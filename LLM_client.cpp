#include "LLM_client.h"

LLMClient::LLMClient(LLM* llm_): llm(llm_)
{
    mode = LOCAL;
    stringWrapper = new StringWrapper();
}

LLMClient::LLMClient(LLMLib* llmLib) : LLMClient((LLM*) llmLib->llm) { }

LLMClient::LLMClient(const std::string& url_, int port_) : url(url_), port(port_) {
    mode = REMOTE;
    stringWrapper = new StringWrapper();
}

//================ Remote requests ================//

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string LLMClient::post_request(const std::string& url, int port, const std::string& path, const std::string& payload) {
    CURL* curl = curl_easy_init();
    std::string response;

    if (curl) {
        std::ostringstream full_url;
        full_url << url << ":" << port << "/" << path;
        std::cout << full_url.str() << std::endl;

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, full_url.str().c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }else
        std::cout << "full_url" << std::endl;

    return response;
}

//================ LLM ================//

std::string LLMClient::handle_tokenize_json(const json& data)
{
    switch (mode) {
    case LOCAL:
        return llm->handle_tokenize_json(data);
    case REMOTE:
        return post_request(url, port, "tokenize", data);
    default:
        std::cerr << "Unknown LLM type" << std::endl;
        return "";
    }
}

std::string LLMClient::handle_detokenize_json(const json& data)
{
    return post_request(url, port, "detokenize", data);
}

std::string LLMClient::handle_embeddings_json(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return post_request(url, port, "embeddings", data);
}

std::string LLMClient::handle_lora_adapters_apply_json(const json& data, httplib::Response* res)
{
    throw std::exception("handle_lora_adapters_apply_json is not supported in remote client");
}

std::string LLMClient::handle_lora_adapters_list_json()
{
    throw std::exception("handle_lora_adapters_list_json is not supported in remote client");
}

std::string LLMClient::handle_completions_json(const json& data, StringWrapper* stringWrapper, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return post_request(url, port, "completion", data);
}

std::string LLMClient::handle_slots_action_json(const json& data, httplib::Response* res)
{
    return post_request(url, port, "slots", data);
}

void LLMClient::handle_cancel_action(int id_slot)
{
    throw std::exception("handle_cancel_action is not supported in remote client");
}