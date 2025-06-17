#include "error_handling.h"

void fail(std::string message, int code) {
    status = code;
    status_msg = message;
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

void init_status() {
    status = 0;
    status_msg.clear();
}

void clear_status() {
    if (status < 0) init_status();
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
    init_status();

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
    init_status();

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
    siglongjmp(sigjmp_buf_point, 1);
}

void sigint_signal_handler(int sig) {
    std::lock_guard<std::mutex> lock(sigint_hook_mutex);
    for (auto& hook : sigint_hooks) {
        try {
            hook(sig);
        }
        catch (...) {}
    }
}

void register_sigint_hook(Hook hook) {
    std::lock_guard<std::mutex> lock(sigint_hook_mutex);
    sigint_hooks.push_back(std::move(hook));
}
