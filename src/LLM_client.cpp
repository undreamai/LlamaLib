#include "LLM_client.h"

LLMClient::LLMClient(LLMProvider* llm_): llm(llm_){ }

std::string LLMClient::tokenize_json(const json& data)
{
    return llm->tokenize_json(data);
}

std::string LLMClient::detokenize_json(const json& data)
{
    return llm->detokenize_json(data);
}

std::string LLMClient::embeddings_json(const json& data)
{
    return llm->embeddings_json(data);
}

std::string LLMClient::completion_json(const json& data, CharArrayFn callback, bool callbackWithJSON)
{
    return llm->completion_json(data, callback, callbackWithJSON);
}

std::string LLMClient::slot_json(const json& data)
{
    return llm->slot_json(data);
}

void LLMClient::cancel(int id_slot)
{
    llm->cancel(id_slot);
}

//================ Remote requests ================//

LLMRemoteClient::LLMRemoteClient(const std::string& url_, const int port_) : url(url_), port(port_) { }

X509_STORE* LLMRemoteClient::load_cert(const std::string& cert_str)
{
  BIO* mem = BIO_new_mem_buf(cert_str.data(), (int) cert_str.size());
  if (!mem) { return nullptr; }

  auto inf = PEM_X509_INFO_read_bio(mem, nullptr, nullptr, nullptr);
  if (!inf) { return nullptr; }

  auto cts = X509_STORE_new();
  if (cts) {
    for (auto i = 0; i < static_cast<int>(sk_X509_INFO_num(inf)); i++) {
      auto itmp = sk_X509_INFO_value(inf, i);
      if (!itmp) { continue; }

      if (itmp->x509) { X509_STORE_add_cert(cts, itmp->x509); }
      if (itmp->crl) { X509_STORE_add_crl(cts, itmp->crl); }
    }
  }

  sk_X509_INFO_pop_free(inf, X509_INFO_free);
  BIO_free(mem);
  return cts;
}

void LLMRemoteClient::set_SSL(const char* SSL_cert){
    this->SSL_cert = SSL_cert;
}

std::string LLMRemoteClient::post_request(
    const std::string& path,
    const json& payload,
    CharArrayFn callback,
    bool callbackWithJSON
) {
    StreamingContext context;
    context.callback = callback;

    json body = payload;
    bool stream = callback != nullptr;
    if (body.contains("stream")) stream = body["stream"];
    else body["stream"] = stream;

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
        {"Accept", stream? "text/event-stream": "application/json"},
        {"Cache-Control", "no-cache"}
    };

    httplib::Request req;
    req.method = "POST";
    req.path = "/" + path;
    req.headers = headers;
    req.body = body.dump();

    std::string concat_string = "";
    std::vector<int> concat_tokens;
    req.content_receiver = [&](const char* data, size_t data_length, uint64_t /*offset*/, uint64_t /*total_length*/) {
        if(stream)
        {
            std::string chunk_str(data, data_length);
            // remove preceding "data: "
            const std::string prefix = "data: ";
            if (chunk_str.rfind(prefix, 0) == 0) {
                chunk_str.erase(0, prefix.length());
            }
            // Remove any trailing newlines or carriage returns
            while (!chunk_str.empty() && (chunk_str.back() == '\n' || chunk_str.back() == '\r')) {
                chunk_str.pop_back();
            }

            json data_json = json::parse(chunk_str);
            if (context.callback != nullptr) {
                if(callbackWithJSON)
                {
                    context.callback(chunk_str.c_str());
                }
                else
                {
                    if (data_json.contains("content"))
                        context.callback(data_json["content"].get<std::string>().c_str());
                }
            }

            if (data_json.contains("content")) {
                concat_string += data_json["content"].get<std::string>();
            }
            if (data_json.contains("tokens")) {
                for (const auto& tok : data_json["tokens"]) {
                    concat_tokens.push_back(tok.get<int>());
                }
            }

            json concat_data = data_json;
            concat_data["content"] = concat_string;
            concat_data["tokens"] = concat_tokens;
            context.buffer = concat_data.dump();
        }
        else
        {
            std::string chunk_str(data, data_length);
            context.buffer += chunk_str;
        }
        return true;
    };

    bool ok = false;

    if (url.rfind("https://", 0) == 0) {
        std::string host = url.substr(8);
        httplib::SSLClient cli(host.c_str(), port);
        if(SSL_cert != "") cli.set_ca_cert_store(load_cert(SSL_cert));
        else cli.enable_server_certificate_verification(false);
        ok = cli.send(req);
    } else {
        std::string host = url.rfind("http://", 0) == 0 ? url.substr(7) : url;
        httplib::Client cli(host.c_str(), port);
        ok = cli.send(req);
    }

    if (!ok) {
        std::string error = "HTTP POST streaming request failed";
        std::cerr << error << std::endl;
        context.buffer = error;
    }

    return context.buffer;
}

std::string LLMRemoteClient::tokenize_json(const json& data)
{
    return post_request("tokenize", data);
}

std::string LLMRemoteClient::detokenize_json(const json& data)
{
    return post_request("detokenize", data);
}

std::string LLMRemoteClient::embeddings_json(const json& data)
{
    return post_request("embeddings", data);
}

std::string LLMRemoteClient::completion_json(const json& data, CharArrayFn callback, bool callbackWithJSON)
{
    return post_request("completion", data, callback, callbackWithJSON);
}


//================ API ================//

LLMClient* LLMClient_Construct(LLMProvider* llm)
{
    return new LLMClient(llm);
}

LLMRemoteClient* LLMRemoteClient_Construct(const char* url, const int port)
{
    return new LLMRemoteClient(url, port);
}