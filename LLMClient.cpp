#include "LLMClient.h"

RemoteLLMClient::RemoteLLMClient(const std::string& url_, int port_)
    : url(url_), port(port_) {
}

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string RemoteLLMClient::post_request(const std::string& url, int port, const std::string& path, const std::string& payload) {
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
