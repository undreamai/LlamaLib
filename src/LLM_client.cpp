#include "LLM_client.h"

//================ Remote requests ================//

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

bool LLMClient::is_server_alive()
{
    if (!is_remote()) return true;

    httplib::Headers headers;
    if (!API_key.empty())
        headers.insert({"Authorization", "Bearer " + API_key});
    auto res = use_ssl ? sslClient->Post("/health", headers) : client->Post("/health", headers);
    return res && res->status >= 200 && res->status < 300;
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

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
        {"Accept", stream ? "text/event-stream" : "application/json"},
        {"Cache-Control", "no-cache"}};
    if (!API_key.empty())
        headers.insert({"Authorization", "Bearer " + API_key});

    httplib::Request req;
    req.method = "POST";
    req.path = "/" + path;
    req.headers = headers;
    req.body = body.dump();

    std::string response_buffer = "";
    ResponseConcatenator concatenator;
    if (stream && callback) concatenator.set_callback(callback, callbackWithJSON);

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

    if (use_ssl)
    {
        sslClient = new httplib::SSLClient(host.c_str(), port);
    }
    else
    {
        client = new httplib::Client(host.c_str(), port);
    }
}

LLMClient::~LLMClient()
{
    if (client != nullptr)
        delete client;
    if (sslClient != nullptr)
        delete sslClient;
}

void LLMClient::set_SSL(const char *SSL_cert_)
{
    if (is_remote())
    {
        this->SSL_cert = SSL_cert_;
        if (sslClient != nullptr)
            sslClient->set_ca_cert_store(load_client_cert(SSL_cert));
    }
}

std::vector<int> LLMClient::tokenize(const std::string &query)
{
    if (is_remote())
    {
        return parse_tokenize_json(json::parse(
            post_request("tokenize", build_tokenize_json(query))));
    }
    else
    {
        return llm->tokenize(query);
    }
}

std::string LLMClient::detokenize(const std::vector<int32_t> &tokens)
{
    if (is_remote())
    {
        return parse_detokenize_json(json::parse(
            post_request("detokenize", build_detokenize_json(tokens))));
    }
    else
    {
        return llm->detokenize(tokens);
    }
}

std::vector<float> LLMClient::embeddings(const std::string &query)
{
    if (is_remote())
    {
        return parse_embeddings_json(json::parse(
            post_request("v1/embeddings", build_embeddings_json(query))));
    }
    else
    {
        return llm->embeddings(query);
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

std::string LLMClient::apply_template(const json &messages)
{
    if (is_remote())
    {
        return parse_apply_template_json(json::parse(
            post_request("apply-template", build_apply_template_json(messages))));
    }
    else
    {
        return llm->apply_template(messages);
    }
}

std::string LLMClient::slot(int id_slot, const std::string &action, const std::string &filepath)
{
    if (is_remote())
    {
        std::cerr << "Slot operations are not supported in remote clients" << std::endl;
        return "";
    }
    else
    {
        if (action == "save")
        {
            return llm->save_slot(id_slot, filepath);
        }
        else if (action == "restore")
        {
            return llm->load_slot(id_slot, filepath);
        }
        else
        {
            throw std::runtime_error("Invalid action" + action);
        }
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