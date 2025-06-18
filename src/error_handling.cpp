#include "error_handling.h"

int& get_status_code() {
    static int status_code = 0;
    return status_code;
}

std::string& get_status_message() {
    static std::string status_message = "";
    return status_message;
}

sigjmp_buf& get_sigjmp_buf_point() {
    static sigjmp_buf sigjmp_buf_point;
    return sigjmp_buf_point;
}

std::mutex& get_sigint_hook_mutex() {
    static std::mutex sigint_hook_mutex;
    return sigint_hook_mutex;
}

std::vector<Hook>& get_sigint_hooks() {
    static std::vector<Hook> sigint_hooks;
    std::lock_guard<std::mutex> lock(get_sigint_hook_mutex());
    return sigint_hooks;
}

void fail(std::string message, int code) {
    int& status_code = get_status_code();
    std::string& status_message = get_status_message();
    status_code = code;
    status_message = message;
}

void handle_exception(int code) {
    try {
        throw;
    }
    catch (const std::exception& ex) {
        fail(ex.what(), code);
    }
    catch (...) {
        fail("Caught unknown exception", code);
    }
}

sigjmp_buf& get_jump_point(bool clear_status) {
    sigjmp_buf& sigjmp_buf_point = get_sigjmp_buf_point();
    if (clear_status)
    {
        int& status_code = get_status_code();
        std::string& status_message = get_status_message();
        status_code = 0;
        status_message.clear();
    }
    return sigjmp_buf_point;
}

static void handle_terminate() {
    crash_signal_handler(1);
}

#ifdef _WIN32

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_CLOSE_EVENT) {
        sigint_signal_handler(SIGINT);
        return TRUE;
    }
    return FALSE;
}

void set_error_handlers(bool crash_handlers, bool sigint_handlers) {
    get_status_code();
    get_status_message();

    if (crash_handlers)
    {
        signal(SIGSEGV, crash_signal_handler);
        signal(SIGFPE, crash_signal_handler);
        std::set_terminate(handle_terminate);
    }

    if (sigint_handlers)
    {
        signal(SIGINT, sigint_signal_handler);
        signal(SIGTERM, sigint_signal_handler);
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    }
}

#else

void handle_signal(int sig, siginfo_t*, void*) {
    crash_signal_handler(sig);
}

void set_error_handlers(bool crash_handlers, bool sigint_handlers) {
    get_status_code();
    get_status_message();

    if (crash_handlers)
    {
        struct sigaction sa {};
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_NODEFER;
        sa.sa_sigaction = handle_signal;

        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGFPE, &sa, nullptr);

        std::set_terminate(handle_terminate);
    }

    if (sigint_handlers)
    {
        struct sigaction shutdown {};
        shutdown.sa_handler = sigint_signal_handler;
        sigemptyset(&shutdown.sa_mask);
        shutdown.sa_flags = 0;

        sigaction(SIGINT, &shutdown, nullptr);
        sigaction(SIGTERM, &shutdown, nullptr);
    }
}
#endif

void crash_signal_handler(int sig) {
    fail("Severe error occurred", sig);
    sigjmp_buf& sigjmp_buf_point = get_sigjmp_buf_point();
    siglongjmp(sigjmp_buf_point, 1);
}

void sigint_signal_handler(int sig) {
    std::vector<Hook>& sigint_hooks = get_sigint_hooks();
    for (auto& hook : sigint_hooks) {
        try {
            hook(sig);
        }
        catch (...) {}
    }
}

void register_sigint_hook(Hook hook) {
    std::vector<Hook>& sigint_hooks = get_sigint_hooks();
    sigint_hooks.push_back(std::move(hook));
}