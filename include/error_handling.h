#pragma once

#include <string>
#include <mutex>
#include <vector>
#include <setjmp.h>
#include <signal.h>
#include <functional>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#define sigjmp_buf jmp_buf
#define sigsetjmp(jb, savemask) setjmp(jb)
#define siglongjmp longjmp
#endif

using Hook = std::function<void(int)>;

/// @brief Error state container for sharing between libraries
struct ErrorState
{
    int status_code = 0;
    std::string status_message = "";
};

/// @brief Error state registry for managing shared error state
class ErrorStateRegistry
{
public:
    /// @brief Inject a custom error state instance
    /// @param state Custom error state instance to use
    /// @details Allows error state injection when using different dynamic libraries
    static void inject_error_state(ErrorState *state)
    {
        custom_error_state_ = state;
    }

    /// @brief Get the error state instance
    /// @return Reference to the error state instance
    static ErrorState &get_error_state()
    {
        if (custom_error_state_)
            return *custom_error_state_;

        static ErrorState error_state;
        return error_state;
    }

private:
    static ErrorState *custom_error_state_;
};

int &get_status_code();
std::string &get_status_message();
sigjmp_buf &get_sigjmp_buf_point();
std::mutex &get_sigint_hook_mutex();
std::vector<Hook> &get_sigint_hooks();

void fail(std::string message, int code = 1);
void handle_exception(int code = -1);
sigjmp_buf &get_jump_point();
#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type);
#else
void handle_signal(int sig, siginfo_t *, void *);
#endif
void set_error_handlers(bool crash_handlers = true, bool sigint_handlers = true);

void crash_signal_handler(int sig);
void sigint_signal_handler(int sig);
void register_sigint_hook(Hook hook);