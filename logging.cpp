#include "logging.h"

void server_log_callback(const DEBUG_LEVEL level, const char* function, int line, const char* message, const json& extra) {
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

    std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";

    if (level >= DEBUG_LEVEL_SET)
    {
        printf("%s", str.c_str());
        if (logCallback) logCallback(str.c_str());
    }
    fflush(stdout);
}

//=========================== API ===========================//

void SetDebugLevel(DEBUG_LEVEL level)
{
    DEBUG_LEVEL_SET = level;
}

void Logging(CharArrayFn callback)
{
    logCallback = callback;
}

void StopLogging()
{
    logCallback = nullptr;
}

void CharArray_Delete(char* str) {
    delete[] str;
}
