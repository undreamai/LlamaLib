#include "LLM_client.h"

// Constructor for local LLM
LLMClient::LLMClient(LLMProvider* llm_) : llm(llm_) { }

// Constructor for remote LLM
LLMClient::LLMClient(const std::string& url_, const int port_) : url(url_), port(port_) { }


void LLMClient::set_SSL(const char* SSL_cert_) {
    if (is_remote()) {
        this->SSL_cert = SSL_cert_;
    }
}

std::string LLMClient::tokenize_json(const json& data)
{
    if (is_remote()) {
        return post_request("tokenize", data);
    } else {
        return llm->tokenize_json(data);
    }
}

std::string LLMClient::detokenize_json(const json& data)
{
    if (is_remote()) {
        return post_request("detokenize", data);
    } else {
        return llm->detokenize_json(data);
    }
}

std::string LLMClient::embeddings_json(const json& data)
{
    if (is_remote()) {
        return post_request("embeddings", data);
    } else {
        return llm->embeddings_json(data);
    }
}

std::string LLMClient::completion_json(const json& data, CharArrayFn callback, bool callbackWithJSON)
{
    if (is_remote()) {
        return post_request("completion", data, callback, callbackWithJSON);
    } else {
        return llm->completion_json(data, callback, callbackWithJSON);
    }
}

std::string LLMClient::apply_template_json(const json& data)
{
    if (is_remote()) {
        return post_request("apply-template", data);
    } else {
        return llm->apply_template_json(data);
    }
}

std::string LLMClient::get_template_json()
{   
    if (is_remote()) {
        return post_request("get-template", {});
    } else {
        return llm->get_template_json();
    }
}

std::string LLMClient::slot_json(const json& data)
{
    if (is_remote()) {
        std::cerr << "Slot operations are not supported in remote clients" << std::endl;
        return "{}";
    } else {
        return llm->slot_json(data);
    }
}

void LLMClient::cancel_json(const json& data)
{
    if (is_remote()) {
        std::cerr << "Cancel is not supported in remote clients" << std::endl;
        return;
    } else {
        llm->cancel_json(data);
    }
}

//================ Remote requests ================//

X509_STORE* load_client_cert(const std::string& cert_str)
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


std::string LLMClient::post_request(
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

    bool https = false;
    std::string host;
    if (url.rfind("https://", 0) == 0) {
        host = url.substr(8);
        https = true;
    }
    else
    {
        host = url.rfind("http://", 0) == 0 ? url.substr(7) : url;
    }

    bool ok = false;
    if(https && SSL_cert != "")
    {
        httplib::SSLClient cli(host.c_str(), port);
        cli.set_ca_cert_store(load_client_cert(SSL_cert));
        ok = cli.send(req);
    }
    else
    {
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

//================ API ================//

void LLMClient_Set_SSL(LLMClient* llm, const char* SSL_cert)
{
    llm->set_SSL(SSL_cert);
}

LLMClient* LLMClient_Construct(LLMProvider* llm)
{
    return new LLMClient(llm);
}

LLMClient* LLMClient_Construct_Remote(const char* url, const int port)
{
    return new LLMClient(url, port);
}