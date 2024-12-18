

#pragma once

#include "json.hpp"

using json = nlohmann::ordered_json;
StringWrapper* logStringWrapper;

void server_log_callback(const char * level, const char * function, int line, const char * message, const json & extra) {
    json log = json{
        {"timestamp", time(nullptr)},
    };

    log.merge_patch({
        {"level",    level},
        {"function", function},
        {"line",     line},
        {"msg",      message},
    });

    if (!extra.empty()) {
        log.merge_patch(extra);
    }

    std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace);
    printf("%s\n", str.c_str());
    if(level != "DEBUG" && logStringWrapper != nullptr) logStringWrapper->AddContent(str+"\n");
    fflush(stdout);
}

#define LOG_INFO(   MSG, ...) server_log_callback("INFO", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log_callback("WARN", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_ERROR(  MSG, ...) server_log_callback("ERR",  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_DEBUG(  MSG, ...) server_log_callback("DEBUG",  __func__, __LINE__, MSG, __VA_ARGS__)

#define SLT_LOG(level, slot, fmt, ...)                                          \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "slot %12.*s: id %2d | task %d | " fmt,                        \
                 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__);         \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define SLT_INF(slot, fmt, ...) SLT_LOG("INFO", slot, fmt, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) SLT_LOG("WARN", slot, fmt, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) SLT_LOG("ERR", slot, fmt, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) SLT_LOG("DEBUG", slot, fmt, __VA_ARGS__)

#define SRV_LOG(level, fmt, ...)                                                \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "srv  %12.*s: " fmt,                                           \
                 12, __func__,  __VA_ARGS__);                                   \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define SRV_INF(fmt, ...) SRV_LOG("INFO", fmt, __VA_ARGS__)
#define SRV_WRN(fmt, ...) SRV_LOG("WARN", fmt, __VA_ARGS__)
#define SRV_ERR(fmt, ...) SRV_LOG("ERR", fmt, __VA_ARGS__)
#define SRV_DBG(fmt, ...) SRV_LOG("DEBUG", fmt, __VA_ARGS__)

#define QUE_LOG(level, fmt, ...)                                                \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "que  %12.*s: " fmt,                                           \
                 12, __func__,  __VA_ARGS__);                                   \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define QUE_INF(fmt, ...) QUE_LOG("INFO", fmt, __VA_ARGS__)
#define QUE_WRN(fmt, ...) QUE_LOG("WARN", fmt, __VA_ARGS__)
#define QUE_ERR(fmt, ...) QUE_LOG("ERR", fmt, __VA_ARGS__)
#define QUE_DBG(fmt, ...) QUE_LOG("DEBUG", fmt, __VA_ARGS__)


