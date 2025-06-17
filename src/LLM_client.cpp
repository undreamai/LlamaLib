#include "LLM_client.h"

LLMClient::LLMClient(LLMProvider* llm_): llm(llm_){ }

std::string LLMClient::tokenize_impl(const json& data)
{
    return llm->tokenize_json(data);
}

std::string LLMClient::detokenize_impl(const json& data)
{
    return llm->detokenize_json(data);
}

std::string LLMClient::embeddings_impl(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return llm->embeddings_json(data, res, is_connection_closed);
}

std::string LLMClient::completion_impl(const json& data, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return llm->completion_json(data, callback, res, is_connection_closed, oaicompat);
}

std::string LLMClient::slot_impl(const json& data, httplib::Response* res)
{
    return llm->slot_json(data, res);
}

void LLMClient::cancel_impl(int id_slot)
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
    CharArrayFn callback
) {
    httplib::Client cli(url.c_str(), port);

    StreamingContext context;
    context.callback = callback;

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
        {"Accept", "text/event-stream"},
        {"Cache-Control", "no-cache"}
    };

    httplib::Request req;
    req.method = "POST";
    req.path = "/" + path;
    req.headers = headers;
    req.body = payload.dump();

    req.content_receiver = [&](const char* data, size_t data_length, uint64_t /*offset*/, uint64_t /*total_length*/) {
        context.buffer.append(data, data_length);
        if (context.callback != nullptr) {
            std::string chunk_str(data, data_length);
            chunk_str.push_back('\0');
            context.callback(chunk_str.c_str());
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

std::string LLMRemoteClient::tokenize_impl(const json& data)
{
    return post_request("tokenize", data);
}

std::string LLMRemoteClient::detokenize_impl(const json& data)
{
    return post_request("detokenize", data);
}

std::string LLMRemoteClient::embeddings_impl(const json& data, httplib::Response* res, std::function<bool()> is_connection_closed)
{
    return post_request("embeddings", data);
}

std::string LLMRemoteClient::completion_impl(const json& data, CharArrayFn callback, httplib::Response* res, std::function<bool()> is_connection_closed, int oaicompat)
{
    return post_request("completion", data, callback);
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