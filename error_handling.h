#pragma once

#include <string>
#include <mutex>
#include <vector>
#include <setjmp.h>
#include <signal.h>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#define sigjmp_buf jmp_buf
#define sigsetjmp(jb, savemask) setjmp(jb)
#define siglongjmp longjmp
#endif

using Hook = std::function<void(int)>;

static int status;
static std::string status_message;
static sigjmp_buf sigjmp_buf_point;
static std::mutex sigint_hook_mutex;
static std::vector<Hook> sigint_hooks;

void fail(std::string message, int code = 1);
void handle_exception(int code = -1);
void init_status();
void clear_status();
#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type);
#else
void handle_signal(int sig, siginfo_t*, void*);
#endif
void set_error_handlers(bool crash_handlers = true, bool sigint_handlers = true);

void crash_signal_handler(int sig);
void sigint_signal_handler(int sig);
void register_sigint_hook(Hook hook);
