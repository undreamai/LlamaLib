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

std::string LLMClient::post_request(
    const std::string &path,
    const json &payload,
    CharArrayFn callback,
    bool callbackWithJSON)
{
    StreamingContext context;
    context.callback = callback;

    json body = payload;
    bool stream = callback != nullptr;
    if (body.contains("stream"))
        stream = body["stream"];
    else
        body["stream"] = stream;

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

    std::string concat_string = "";
    std::vector<int> concat_tokens;
    req.content_receiver = [&](const char *data, size_t data_length, uint64_t /*offset*/, uint64_t /*total_length*/)
    {
        if (stream)
        {
            std::string chunk_str(data, data_length);
            // remove preceding "data: "
            const std::string prefix = "data: ";
            if (chunk_str.rfind(prefix, 0) == 0)
            {
                chunk_str.erase(0, prefix.length());
            }
            // Remove any trailing newlines or carriage returns
            while (!chunk_str.empty() && (chunk_str.back() == '\n' || chunk_str.back() == '\r'))
            {
                chunk_str.pop_back();
            }

            json data_json = json::parse(chunk_str);
            if (data_json.contains("content"))
            {
                concat_string += data_json["content"].get<std::string>();
            }
            if (data_json.contains("tokens"))
            {
                for (const auto &tok : data_json["tokens"])
                {
                    concat_tokens.push_back(tok.get<int>());
                }
            }

            json concat_data = data_json;
            concat_data["content"] = concat_string;
            concat_data["tokens"] = concat_tokens;

            if (context.callback != nullptr)
            {
                if (callbackWithJSON)
                    context.callback(concat_data.dump().c_str());
                else
                    context.callback(concat_string.c_str());
            }

            context.buffer = concat_data.dump();
        }
        else
        {
            std::string chunk_str(data, data_length);
            context.buffer += chunk_str;
        }
        return true;
    };

    bool ok = use_ssl ? sslClient->send(req) : client->send(req);
    if (!ok)
    {
        std::string error = "HTTP POST streaming request failed";
        std::cerr << error << std::endl;
        context.buffer = error;
    }

    return context.buffer;
}

//================ LLMClient ================//

// Constructor for local LLM
LLMClient::LLMClient(LLMProvider *llm_) : llm(llm_) {}

// Constructor for remote LLM
LLMClient::LLMClient(const std::string &url_, const int port_, const std::string &API_key_) : url(url_), port(port_), API_key(API_key_)
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
            post_request("embeddings", build_embeddings_json(query))));
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

std::string LLMClient::get_template()
{
    if (is_remote())
    {
        return post_request("get-template", {});
    }
    else
    {
        return llm->get_template();
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
        std::cerr << "Cancel is not supported in remote clients" << std::endl;
        return;
    }
    else
    {
        llm->cancel(id_slot);
    }
}

//================ API ================//

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