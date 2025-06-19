#pragma once

#include "defs.h"
#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::ordered_json;
static CharArrayFn logCallback = nullptr;

enum DEBUG_LEVEL {
    DEBUG = 0,
    INFO,
    WARNING,
    ERR
};

static DEBUG_LEVEL DEBUG_LEVEL_SET = INFO;

static inline const char* stringToCharArray(const std::string& input)
{
    char* content = new char[input.length() + 1];
    strcpy(content, input.c_str());
    return content;
}

void server_log_callback(const DEBUG_LEVEL level, const char* function, int line, const char* message, const json& extra);

#define LOG_INFO(   MSG, ...) server_log_callback(INFO, __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log_callback(WARNING, __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_ERROR(  MSG, ...) server_log_callback(ERR,  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_DEBUG(  MSG, ...) server_log_callback(DEBUG,  __func__, __LINE__, MSG, __VA_ARGS__)

#define SLT_LOG(level, slot, fmt, ...)                                          \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "slot %12.*s: id %2d | task %d | " fmt,                        \
                 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__);         \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define SLT_INF(slot, fmt, ...) SLT_LOG(INFO, slot, fmt, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) SLT_LOG(WARNING, slot, fmt, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) SLT_LOG(ERR, slot, fmt, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) SLT_LOG(DEBUG, slot, fmt, __VA_ARGS__)

#define SRV_LOG(level, fmt, ...)                                                \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "srv  %12.*s: " fmt,                                           \
                 12, __func__,  __VA_ARGS__);                                   \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define SRV_INF(fmt, ...) SRV_LOG(INFO, fmt, __VA_ARGS__)
#define SRV_WRN(fmt, ...) SRV_LOG(WARNING, fmt, __VA_ARGS__)
#define SRV_ERR(fmt, ...) SRV_LOG(ERR, fmt, __VA_ARGS__)
#define SRV_DBG(fmt, ...) SRV_LOG(DEBUG, fmt, __VA_ARGS__)

#define QUE_LOG(level, fmt, ...)                                                \
    do {                                                                        \
        char formatted_msg[512];                                                \
        snprintf(formatted_msg, sizeof(formatted_msg),                          \
                 "que  %12.*s: " fmt,                                           \
                 12, __func__,  __VA_ARGS__);                                   \
        server_log_callback(level, __func__, __LINE__, formatted_msg, json{});  \
    } while (0)

#define QUE_INF(fmt, ...) QUE_LOG(INFO, fmt, __VA_ARGS__)
#define QUE_WRN(fmt, ...) QUE_LOG(WARNING, fmt, __VA_ARGS__)
#define QUE_ERR(fmt, ...) QUE_LOG(ERR, fmt, __VA_ARGS__)
#define QUE_DBG(fmt, ...) QUE_LOG(DEBUG, fmt, __VA_ARGS__)

extern "C" {
    UNDREAMAI_API void LLM_Debug(DEBUG_LEVEL level);
    UNDREAMAI_API void LLM_Logging_Callback(CharArrayFn callback);
    UNDREAMAI_API void LLM_Logging_Stop();
    UNDREAMAI_API void CharArray_Delete(char* object);
#ifdef _DEBUG
    UNDREAMAI_API const bool IsDebuggerAttached(void);
#endif
}
