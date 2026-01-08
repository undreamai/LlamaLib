#include "LLM_client.h"

//================ Remote requests ================//

#if !(TARGET_OS_IOS || TARGET_OS_VISION)
X509_STORE *load_client_cert(const std::string &cert_str)
{
    BIO *mem = BIO_new_mem_buf(cert_str.data(), (int)cert_str.size());
    if (!mem)
    {
        return nullptr;
    }

    auto inf = PEM_X509_INFO_read_bio(mem, nullptr, nullptr, nullptr);
    if (!inf)
    {
        return nullptr;
    }

    auto cts = X509_STORE_new();
    if (cts)
    {
        for (auto i = 0; i < static_cast<int>(sk_X509_INFO_num(inf)); i++)
        {
            auto itmp = sk_X509_INFO_value(inf, i);
            if (!itmp)
            {
                continue;
            }

            if (itmp->x509)
            {
                X509_STORE_add_cert(cts, itmp->x509);
            }
            if (itmp->crl)
            {
                X509_STORE_add_crl(cts, itmp->crl);
            }
        }
    }

    sk_X509_INFO_pop_free(inf, X509_INFO_free);
    BIO_free(mem);
    return cts;
}
#endif

bool LLMClient::is_server_alive()
{
    if (!is_remote()) return true;

    std::vector<std::pair<std::string, std::string>> headers;
    if (!API_key.empty()) {
        headers.push_back({"Authorization", "Bearer " + API_key});
    }

#if TARGET_OS_IOS || TARGET_OS_VISION
    HttpResult result = transport->post_request("health", "{}", headers);
    return result.success && result.status_code >= 200 && result.status_code < 300;
#else
    httplib::Headers Headers;
    for (const auto& h : headers) Headers.insert(h);
    auto res = use_ssl ? sslClient->Post("/health", Headers) : client->Post("/health", Headers);
    return res && res->status >= 200 && res->status < 300;
#endif
}

std::string LLMClient::post_request(
    const std::string &path,
    const json &payload,
    CharArrayFn callback,
    bool callbackWithJSON)
{
    json body = payload;
    bool stream = callback != nullptr;
    if (body.contains("stream"))
        stream = body["stream"];
    else
        body["stream"] = stream;

    bool* cancel_flag = new bool(false);
    if (stream) active_requests.push_back(cancel_flag);

    std::string response_buffer = "";
    ResponseConcatenator concatenator;
    if (stream && callback) concatenator.set_callback(callback, callbackWithJSON);

    std::vector<std::pair<std::string, std::string>> headers = {
        {"Content-Type", "application/json"},
        {"Accept", stream ? "text/event-stream" : "application/json"},
        {"Cache-Control", "no-cache"}
    };
    
    if (!API_key.empty()) {
        headers.push_back({"Authorization", "Bearer " + API_key});
    }

#if TARGET_OS_IOS || TARGET_OS_VISION
    // iOS Native Implementation
    CharArrayFn ios_callback;
    if (stream) {
        ios_callback = [&](const char* data, size_t length) -> bool {
            std::string chunk_str(data, length);
            if (!concatenator.process_chunk(chunk_str)) {
                return false;
            }
            if (*cancel_flag) {
                std::cerr << "[LLMClient] Streaming cancelled\n";
                return false;
            }
            return true;
        };
    }

    HttpResult result;
    for (int attempt = 0; attempt <= max_retries; attempt++) {
        result = transport->post_request(path, body.dump(), headers, ios_callback, cancel_flag);

        if (result.success || *cancel_flag) break;

        int delay_seconds = std::min(30, 1 << attempt);
        std::cerr << "[LLMClient] POST failed: " << result.error_message
                  << ", retrying in " << delay_seconds << "s (attempt "
                  << attempt << "/" << max_retries << ")\n";
        std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
    }

    if (!result.success) {
        std::cerr << "[LLMClient] POST request failed: " << result.error_message << "\n";
        if (stream) {
            active_requests.erase(std::remove(active_requests.begin(), active_requests.end(), cancel_flag), active_requests.end());
        }
        delete cancel_flag;
        return "{}";
    }

    if (stream) {
        active_requests.erase(std::remove(active_requests.begin(), active_requests.end(), cancel_flag), active_requests.end());
    }
    delete cancel_flag;

    return stream ? concatenator.get_result_json() : result.body;
    
#else
    httplib::Headers Headers;
    for (const auto& h : headers) Headers.insert(h);

    httplib::Request req;
    req.method = "POST";
    req.path = "/" + path;
    req.headers = Headers;
    req.body = body.dump();

    req.content_receiver = [&](const char *data, size_t data_length, uint64_t /*offset*/, uint64_t /*total_length*/)
    {
        std::string chunk_str(data, data_length);
        if (stream)
        {
            if (!concatenator.process_chunk(chunk_str)) {
                return false;
            }
            if (*cancel_flag)
            {
                std::cerr << "[LLMClient] Streaming cancelled\n";
                return false;
            }
        }
        else
        {
            response_buffer += chunk_str;
        }
        return true;
    };

    const int max_delay = 30;
    bool request_sent;
    for (int attempt = 0; attempt <= max_retries; attempt++)
    {
        request_sent = use_ssl ? sslClient->send(req) : client->send(req);
        if (request_sent || *cancel_flag) break;

        int delay_seconds = std::min(max_delay, 1 << attempt);
        std::cerr << "[LLMClient] POST failed, retrying in " << delay_seconds
                  << "s (attempt " << attempt << "/" << max_retries << ")\n";
        std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
    }

    if (!request_sent)
    {
        std::cerr << "[LLMClient] POST request failed after retries\n";
        return "{}";
    }

    if (stream) active_requests.erase(std::remove(active_requests.begin(), active_requests.end(), cancel_flag), active_requests.end());
    delete cancel_flag;

    if (stream) {
        return concatenator.get_result_json();
    } else {
        return response_buffer;
    }
#endif
}

//================ LLMClient ================//

// Constructor for local LLM
LLMClient::LLMClient(LLMProvider *llm_) : llm(llm_) {}

// Constructor for remote LLM
LLMClient::LLMClient(const std::string &url_, const int port_, const std::string &API_key_, const int max_retries_) : url(url_), port(port_), API_key(API_key_), max_retries(max_retries_)
{
    std::string host;
    if (url.rfind("https://", 0) == 0)
    {
        host = url.substr(8);
        use_ssl = true;
    }
    else
    {
        host = url.rfind("http://", 0) == 0 ? url.substr(7) : url;
        use_ssl = false;
    }

#if TARGET_OS_IOS || TARGET_OS_VISION
    transport = new IOSHttpTransport(host, use_ssl, port);
    transport->set_timeout(60.0);
#else
    if (use_ssl)
    {
        sslClient = new httplib::SSLClient(host.c_str(), port);
    }
    else
    {
        client = new httplib::Client(host.c_str(), port);
    }
#endif
}

LLMClient::~LLMClient()
{
#if TARGET_OS_IOS || TARGET_OS_VISION
    if (transport != nullptr) {
        delete transport;
    }
#else
    if (client != nullptr)
        delete client;
    if (sslClient != nullptr)
        delete sslClient;
#endif
}

void LLMClient::set_SSL(const char *SSL_cert_)
{
#if !(TARGET_OS_IOS || TARGET_OS_VISION)
    if (is_remote())
    {
        this->SSL_cert = SSL_cert_;
        if (sslClient != nullptr)
            sslClient->set_ca_cert_store(load_client_cert(SSL_cert));
    }
#endif
}

std::string LLMClient::tokenize_json(const json &data)
{
    if (is_remote())
    {
        return post_request("tokenize", data);
    }
    else
    {
        return llm->tokenize_json(data);
    }
}

std::string LLMClient::detokenize_json(const json &data)
{
    if (is_remote())
    {
        return post_request("detokenize", data);
    }
    else
    {
        return llm->detokenize_json(data);
    }
}

std::string LLMClient::embeddings_json(const json &data)
{
    if (is_remote())
    {
        return post_request("embeddings", data);
    }
    else
    {
        return llm->embeddings_json(data);
    }
}

std::string LLMClient::completion_json(const json &data, CharArrayFn callback, bool callbackWithJSON)
{
    if (is_remote())
    {
        json data_remote = data;
        if (data.contains("id_slot") && data["id_slot"] != -1)
        {
            std::cerr << "Remote clients can only use id_slot -1" << std::endl;
            data_remote["id_slot"] = -1;
        }
        return post_request("completion", data_remote, callback, callbackWithJSON);
    }
    else
    {
        return llm->completion_json(data, callback, callbackWithJSON);
    }
}

int LLMClient::get_next_available_slot()
{
    if (is_remote())
        return -1;
    return llm->get_next_available_slot();
}

std::string LLMClient::apply_template_json(const json &data)
{
    if (is_remote())
    {
        return post_request("apply-template", data);
    }
    else
    {
        return llm->apply_template_json(data);
    }
}

std::string LLMClient::slot_json(const json &data)
{
    if (is_remote())
    {
        std::cerr << "Slot operations are not supported in remote clients" << std::endl;
        return "{}";
    }
    else
    {
        return llm->slot_json(data);
    }
}

void LLMClient::cancel(int id_slot)
{
    if (is_remote())
    {
        for (bool* flag : active_requests) *flag = true;
    }
    else
    {
        llm->cancel(id_slot);
    }
}

//================ API ================//

bool LLMClient_Is_Server_Alive(LLMClient *llm)
{
    return llm->is_server_alive();
}

void LLMClient_Set_SSL(LLMClient *llm, const char *SSL_cert)
{
    llm->set_SSL(SSL_cert);
}

LLMClient *LLMClient_Construct(LLMProvider *llm)
{
    return new LLMClient(llm);
}

LLMClient *LLMClient_Construct_Remote(const char *url, const int port, const char *API_key)
{
    return new LLMClient(url, port, API_key);
}