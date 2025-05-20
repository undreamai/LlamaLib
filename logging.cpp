#include "logging.h"

void server_log_callback(const char* level, const char* function, int line, const char* message, const json& extra) {
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

    if (level != "DEBUG")
    {
        printf("%s\n", str.c_str());
        if (logStringWrapper != nullptr) logStringWrapper->AddContent(str + "\n");
    }
    fflush(stdout);
}

//=========================== API ===========================//

const void Logging(StringWrapper* wrapper)
{
    logStringWrapper = wrapper;
}

const void StopLogging()
{
    logStringWrapper = nullptr;
}
