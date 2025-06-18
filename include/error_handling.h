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

int& get_status_code();
std::string& get_status_message();
sigjmp_buf& get_sigjmp_buf_point();
std::mutex& get_sigint_hook_mutex();
std::vector<Hook>& get_sigint_hooks();

void fail(std::string message, int code = 1);
void handle_exception(int code = -1);
sigjmp_buf& get_jump_point(bool clear_status=false);
#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type);
#else
void handle_signal(int sig, siginfo_t*, void*);
#endif
void set_error_handlers(bool crash_handlers = true, bool sigint_handlers = true);

void crash_signal_handler(int sig);
void sigint_signal_handler(int sig);
void register_sigint_hook(Hook hook);
