#pragma once
#include "utils.hpp"
#include "stringwrapper.h"

#undef LOG_ERROR
#undef LOG_WARNING
#undef LOG_INFO

#define LOG_ERROR(  MSG, ...) server_log_callback("ERR",  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log_callback("WARN", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(   MSG, ...) server_log_callback("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

StringWrapper* logStringWrapper;

static inline void server_log_callback(const char * level, const char * function, int line, const char * message, const json & extra) {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = json{
        {"tid",       ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    std::string str;
    if (server_log_json) {
        log.merge_patch({
            {"level",    level},
            {"function", function},
            {"line",     line},
            {"msg",      message},
        });

        if (!extra.empty()) {
            log.merge_patch(extra);
        }

        str = log.dump(-1, ' ', false, json::error_handler_t::replace);
        printf("%s\n", str.c_str());
    } else {
        char buf[1024];
        snprintf(buf, 1024, "%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto& el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            ss << " " << el.key() << "=" << value;
        }

        str = ss.str();
        printf("%.*s\n", (int)str.size(), str.data());
    }
    if(logStringWrapper != nullptr) logStringWrapper->AddContent(str+"\n");
    fflush(stdout);
}