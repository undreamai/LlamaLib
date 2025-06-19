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

#ifdef _DEBUG
#if (defined __linux__)
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>

static bool debuggerIsAttached()
{
    char buf[4096];

    const int status_fd = open("/proc/self/status", O_RDONLY);
    if (status_fd == -1)
        return false;

    const ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);
    close(status_fd);

    if (num_read <= 0)
        return false;

    buf[num_read] = '\0';
    constexpr char tracerPidString[] = "TracerPid:";
    const auto tracer_pid_ptr = strstr(buf, tracerPidString);
    if (!tracer_pid_ptr)
        return false;

    for (const char* characterPtr = tracer_pid_ptr + sizeof(tracerPidString) - 1; characterPtr <= buf + num_read; ++characterPtr)
    {
        if (isspace(*characterPtr))
            continue;

        return isdigit(*characterPtr) != 0 && *characterPtr != '0';
    }

    return false;
}
#endif 

#ifdef __APPLE__
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/sysctl.h>

static bool AmIBeingDebugged(void)
// Returns true if the current process is being debugged (either 
// running under the debugger or has a debugger attached post facto).
{
    int                 junk;
    int                 mib[4];
    struct kinfo_proc   info;
    size_t              size;

    // Initialize the flags so that, if sysctl fails for some bizarre 
    // reason, we get a predictable result.

    info.kp_proc.p_flag = 0;

    // Initialize mib, which tells sysctl the info we want, in this case
    // we're looking for information about a specific process ID.

    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();

    // Call sysctl.

    size = sizeof(info);
    junk = sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);
    assert(junk == 0);

    // We're being debugged if the P_TRACED flag is set.

    return ((info.kp_proc.p_flag & P_TRACED) != 0);
}
#endif
#endif

//=========================== API ===========================//

void LLM_Debug(DEBUG_LEVEL level)
{
    DEBUG_LEVEL_SET = level;
}

void LLM_Logging_Callback(CharArrayFn callback)
{
    logCallback = callback;
}

void LLM_Logging_Stop()
{
    logCallback = nullptr;
}

void CharArray_Delete(char* str) {
    delete[] str;
}

#ifdef _DEBUG
const bool IsDebuggerAttached(void) {
#ifdef _MSC_VER
    return ::IsDebuggerPresent();
#elif __APPLE__
    return AmIBeingDebugged();
#elif __linux__
    return debuggerIsAttached();
#else
    return false;
#endif
}
#endif