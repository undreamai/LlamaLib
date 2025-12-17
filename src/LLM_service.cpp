#include "LLM_service.h"

#include "common.h"
#include "llama-chat.h"
#include "log.h"

#ifndef SERVER_H
#define SERVER_H
#include "server-context.cpp"
#endif // SERVER_H

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

    for (const char *characterPtr = tracer_pid_ptr + sizeof(tracerPidString) - 1; characterPtr <= buf + num_read; ++characterPtr)
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
    int junk;
    int mib[4];
    struct kinfo_proc info;
    size_t size;

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

//============================= LLMService IMPLEMENTATION =============================//

LLMService::LLMService() {}

LLMService::LLMService(const std::string &model_path, int num_slots, int num_threads, int num_GPU_layers, bool flash_attention, int context_size, int batch_size, bool embedding_only, const std::vector<std::string> &lora_paths)
{
    init(LLM::LLM_args_to_command(model_path, num_slots, num_threads, num_GPU_layers, flash_attention, context_size, batch_size, embedding_only, lora_paths));
}

LLMService *LLMService::from_params(const json &params_json)
{
    std::vector<char *> argv = LLMService::jsonToArguments(params_json);
    LLMService *llmService = new LLMService();
    llmService->init(argv.size(), argv.data());
    return llmService;
}

LLMService *LLMService::from_command(const std::string &command)
{
    LLMService *llmService = new LLMService();
    llmService->init(command);
    return llmService;
}

LLMService *LLMService::from_command(int argc, char **argv)
{
    LLMService *llmService = new LLMService();
    llmService->init(argc, argv);
    return llmService;
}

LLMService::~LLMService()
{
    stop_server();
    stop();
    if (ctx_server != nullptr)
    {
        delete ctx_server;
        ctx_server = nullptr;
    }
}

std::vector<char *> LLMService::jsonToArguments(const json &params_json)
{
    common_params default_params;
    common_params_context ctx = common_params_parser_init(default_params, LLAMA_EXAMPLE_SERVER);

    std::vector<std::string> args_str = {"llm"};
    std::set<std::string> used_keys;

    for (const auto &opt : ctx.options)
    {
        for (const auto &name : opt.args)
        {
            std::string key = name;
            if (key.rfind("--", 0) == 0)
                key = key.substr(2); // strip leading "--"
            else if (key.rfind("-", 0) == 0)
                continue; // skip short options

            std::string json_key = key;
            std::replace(json_key.begin(), json_key.end(), '-', '_');

            if (params_json.contains(json_key))
                continue;

            used_keys.insert(json_key);
            const auto &value = params_json[json_key];
            args_str.push_back(name);

            if (opt.handler_void != nullptr)
            {
                break;
            }
            else if (opt.handler_string != nullptr || opt.handler_int != nullptr)
            {
                args_str.push_back(value.is_string() ? value.get<std::string>() : value.dump());
                break;
            }
            else if (opt.handler_str_str != nullptr)
            {
                if (!value.is_array() || value.size() != 2)
                {
                    std::string err = "Expected array of 2 values for: " + json_key;
                    LOG_WRN("%s\n", err.c_str());
                    continue;
                }
                args_str.push_back(value[0].is_string() ? value[0].get<std::string>() : value[0].dump());
                args_str.push_back(value[1].is_string() ? value[1].get<std::string>() : value[1].dump());
                break;
            }
        }
    }

    // Report unused keys
    for (const auto &[key, _] : params_json.items())
    {
        if (used_keys.find(key) == used_keys.end())
        {
            std::string err = "Unused parameter in JSON: " + key;
            LOG_WRN("%s\n", err.c_str());
        }
    }

    // Convert to argv
    std::vector<std::unique_ptr<char[]>> argv_storage;
    std::vector<char *> argv;
    for (const auto &arg : args_str)
    {
        auto buf = std::make_unique<char[]>(arg.size() + 1);
        std::memcpy(buf.get(), arg.c_str(), arg.size() + 1);
        argv.push_back(buf.get());
        argv_storage.push_back(std::move(buf));
    }

    return argv;
}

std::vector<std::string> LLMService::splitArguments(const std::string &inputString)
{
    std::vector<std::string> arguments;

    unsigned counter = 0;
    std::string segment;
    std::istringstream stream_input(inputString);
    while (std::getline(stream_input, segment, '"'))
    {
        ++counter;
        if (counter % 2 == 0)
        {
            if (!segment.empty())
                arguments.push_back(segment);
        }
        else
        {
            std::istringstream stream_segment(segment);
            while (std::getline(stream_segment, segment, ' '))
                if (!segment.empty())
                    arguments.push_back(segment);
        }
    }
    return arguments;
}

void LLMService::init(const std::string &params_string)
{
    std::vector<std::string> arguments = splitArguments("llm " + params_string);

    // Convert vector of strings to argc and argv
    int argc = static_cast<int>(arguments.size());
    char **argv = new char *[argc];
    for (int i = 0; i < argc; ++i)
    {
        argv[i] = new char[arguments[i].size() + 1];
        std::strcpy(argv[i], arguments[i].c_str());
    }
    init(argc, argv);
}

void LLMService::init(const char *params_string)
{
    init(std::string(params_string));
}

void LLMService::init(int argc, char **argv)
{
    ensure_error_handlers_initialized();
    if (setjmp(get_jump_point()) != 0)
        return;
    try
    {
        command = args_to_command(argc, argv);

        LLMProviderRegistry &registry = LLMProviderRegistry::instance();
        registry.register_instance(this);
        debug(registry.get_debug_level());
        logging_callback(registry.get_log_callback());

        ctx_server = new server_context();
        // ctx_server->impl->batch = {0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        params = new common_params();
        params->port = 0;
        if (!common_params_parse(argc, argv, *params, LLAMA_EXAMPLE_SERVER))
        {
            throw std::runtime_error("Invalid parameters!");
        }

        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params->embedding && params->n_batch > params->n_ubatch) {
            LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params->n_batch, params->n_ubatch);
            LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params->n_ubatch);
            params->n_batch = params->n_ubatch;
        }

        // for consistency between server router mode and single-model mode, we set the same model name as alias
        if (params->model_alias.empty() && !params->model.name.empty()) {
            params->model_alias = params->model.name;
        }

        common_init();

        llama_backend_init();
        llama_backend_has_init = true;
        llama_numa_init(params->numa);

        LLAMALIB_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params->cpuparams.n_threads, params->cpuparams_batch.n_threads, std::thread::hardware_concurrency());

        // load the model
        if (!ctx_server->load_model(*params))
        {
            throw std::runtime_error("Error loading the model!");
        }
        ctx_server->impl->params_base.use_jinja = true;
        ctx_server->init();
        enable_reasoning(reasoning_enabled);
        LLAMALIB_INF("model loaded\n");

        ctx_http = new server_http_context();
        routes = new server_routes(*params, *ctx_server);

        params->chat_template = detect_chat_template();
        LLAMALIB_INF("chat_template: %s\n", params->chat_template.c_str());

        ctx_server->impl->queue_tasks.on_new_task([this](server_task && task)
                                            { this->ctx_server->impl->process_single_task(std::move(task)); });
        ctx_server->impl->queue_tasks.on_update_slots([this]()
                                                { this->ctx_server->impl->update_slots(); });
    }
    catch (...)
    {
        LLMProviderRegistry::instance().unregister_instance(this);
        handle_exception(1);
    }
}

void LLMService::enable_reasoning(bool reasoning) {
    LLMProvider::enable_reasoning(reasoning);
    if (ctx_server != nullptr) ctx_server->impl->oai_parser_opt.enable_thinking = reasoning_enabled;
}

const std::string LLMService::detect_chat_template()
{
    const char *chat_template_jinja = common_chat_templates_source(ctx_server->impl->chat_templates.get());
    int chat_template_value = llm_chat_detect_template(chat_template_jinja);
    std::vector<const char *> supported_tmpl;
    int res = llama_chat_builtin_templates(nullptr, 0);
    if (res > 0)
    {
        supported_tmpl.resize(res);
        llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
        for (const auto &key : supported_tmpl)
        {
            llm_chat_template val = llm_chat_template_from_str(key);
            if ((int)val == chat_template_value)
            {
                return key;
                break;
            }
        }
    }
    return "";
}

void LLMService::debug(int debug_level)
{
    common_log_set_verbosity_thold(debug_level - 2);
}

void LLMService::logging_callback(CharArrayFn callback)
{
    log_callback = callback;
}

const json handle_post(const httplib::Request &req, httplib::Response &res)
{
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    return json::parse(req.body);
};

void res_ok(httplib::Response &res, std::string data)
{
    res.set_content(data, "application/json; charset=utf-8");
    res.status = 200;
};

void handle_error(httplib::Response &res, const json error_data)
{
    json final_response{{"error", error_data}};
    res.set_content(final_response.dump(), "application/json; charset=utf-8");
    res.status = 500;
}

void release_slot(server_slot &slot)
{
    if (slot.task && slot.task->type == SERVER_TASK_TYPE_COMPLETION)
    {
        slot.i_batch = -1;
        slot.task->params.n_predict = 0;
        slot.stop = STOP_TYPE_LIMIT;
        slot.has_next_token = false;
    }
    else
    {
        slot.release();
    }
}

int LLMService::get_next_available_slot()
{
    if (setjmp(get_jump_point(true)) != 0)
        return -1;
    if (ctx_server->impl->slots.size() == 0)
        return -1;
    return next_available_slot++ % ctx_server->impl->slots.size();
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

void LLMService::start_server(const std::string &host, int port, const std::string &API_key)
{
    if (setjmp(get_jump_point(true)) != 0)
        return;

    try
    {
        params->hostname = host.empty() ? "0.0.0.0" : host;
        if (port >= 0)
            params->port = port;
        params->api_keys.clear();
        if (!API_key.empty())
            params->api_keys.push_back(API_key);

        std::lock_guard<std::mutex> lock(start_stop_mutex);

        if (!ctx_http->init(*params)) {
            throw std::runtime_error("Failed to initialize HTTP server!");
        }

        // register API routes
        ctx_http->post("/health",  ex_wrapper(routes->get_health)); // public endpoint (no API key check)
        ctx_http->post("/v1/health", ex_wrapper(routes->get_health)); // public endpoint (no API key check)
        ctx_http->post("/completion", ex_wrapper(routes->post_completions)); // legacy
        ctx_http->post("/completions", ex_wrapper(routes->post_completions));
        ctx_http->post("/chat/completions", ex_wrapper(routes->post_chat_completions));
        ctx_http->post("/v1/chat/completions", ex_wrapper(routes->post_chat_completions));
        ctx_http->post("/tokenize", ex_wrapper(routes->post_tokenize));
        ctx_http->post("/detokenize", ex_wrapper(routes->post_detokenize));
        ctx_http->post("/apply-template", ex_wrapper(routes->post_apply_template));
        ctx_http->post("/embedding", ex_wrapper(routes->post_embeddings)); // legacy
        ctx_http->post("/embeddings", ex_wrapper(routes->post_embeddings));


        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http->start()) {
            stop();
            throw std::runtime_error("Exiting due to HTTP server error\n");
        }

        ctx_http->is_ready.store(true);
    }
    catch (...)
    {
        handle_exception();
    }
}


void LLMService::stop_server()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    if (ctx_http == nullptr)
        return;
    std::lock_guard<std::mutex> lock(start_stop_mutex);
    LLAMALIB_INF("stopping server\n");
    ctx_http->stop();
    if (ctx_http->thread.joinable()) ctx_http->thread.join();
    server_stopped = true;
    server_stopped_cv.notify_all();
    LLAMALIB_INF("stopped server\n");
}

void LLMService::join_server()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    std::unique_lock<std::mutex> lock(start_stop_mutex);
    server_stopped_cv.wait(lock, [this]
                           { return server_stopped; });
}

void LLMService::start()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    std::lock_guard<std::mutex> lock(start_stop_mutex);
    service_thread = std::thread([&]()
                                 {
        LLAMALIB_INF("starting service\n");
        ctx_server->impl->queue_tasks.start_loop();
        LLAMALIB_INF("stopped service loop\n");
        return 1; });
    while (!started())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void LLMService::stop()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    try
    {
        std::lock_guard<std::mutex> lock(start_stop_mutex);
        if (!started())
            return;
        LLAMALIB_INF("shutting down tasks\n");

        // hack completion slots to think task is completed
        for (server_slot &slot : ctx_server->impl->slots)
        {
            release_slot(slot);
        }

        if((!ctx_server->impl->queue_tasks.is_empty()))
        {
            LLAMALIB_INF("Wait until tasks have finished\n");
            int grace = 20;
            while (!ctx_server->impl->queue_tasks.is_empty() && grace-- > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            LLAMALIB_INF("Tasks have finished\n");
        }

        ctx_http->stop();
        ctx_server->terminate();

        if (llama_backend_has_init)
            llama_backend_free();

        if (service_thread.joinable())
        {
            service_thread.join();
        }
        service_stopped = true;
        service_stopped_cv.notify_all();
        LLAMALIB_INF("service stopped\n");

        LLMProviderRegistry::instance().unregister_instance(this);
    }
    catch (...)
    {
        handle_exception();
    }
}

void LLMService::join_service()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    std::unique_lock<std::mutex> lock(start_stop_mutex);
    service_stopped_cv.wait(lock, [this]
                            { return service_stopped; });
}

bool LLMService::started()
{
    return ctx_server != nullptr && ctx_server->impl->queue_tasks.is_running();
}

void LLMService::set_SSL(const std::string &SSL_cert_str, const std::string &SSL_key_str)
{
    params->ssl_cert = SSL_cert_str;
    params->ssl_key = SSL_key_str;
}

std::string LLMService::encapsulate_route(const json &body, server_http_context::handler_t route_handler)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";

    try
    {
        server_http_req req{ {}, {}, "", body.dump(), always_false };
        return route_handler(req)->data;
    }
    catch (...)
    {
        handle_exception();
    }
    return "";
}

std::string LLMService::apply_template_json(const json &body)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    std::vector<raw_buffer> files; // dummy, unused
    json copy = body;
    json data = oaicompat_chat_params_parse(
        copy,
        ctx_server->impl->oai_parser_opt,
        files);
    return safe_json_to_str({{"prompt", std::move(data.at("prompt"))}});
}

std::string LLMService::tokenize_json(const json &body)
{
    return encapsulate_route(body, routes->post_tokenize);
}

std::string LLMService::detokenize_json(const json &body)
{
    return encapsulate_route(body, routes->post_detokenize);
}

std::string LLMService::embeddings_json(const json &body)
{
    return encapsulate_route(body, routes->post_embeddings);
};

std::string LLMService::lora_weight_json(const json &body)
{
    return safe_json_to_str(encapsulate_route(body, routes->post_lora_adapters));
};

std::string LLMService::lora_list_json()
{
    return encapsulate_route({}, routes->get_lora_adapters);
}

std::string LLMService::completion_json(const json &data_in, CharArrayFn callback, bool callbackWithJSON)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    
    try
    {
        bool stream = json_value(data_in, "stream", callback != nullptr);
        json data = data_in;
        data["stream"] = stream;

        server_http_req req{ {}, {}, "", data.dump(), always_false };
        auto result = routes->post_completions(req);
        if (result->status != 200)
        {
            return result->data;
        }

        if (stream)
        {
            ResponseConcatenator concatenator;
            if (callback) concatenator.set_callback(callback, callbackWithJSON);
            while (!concatenator.is_complete()) {
                std::string chunk;
                bool has_next = result->next(chunk);
                if (!chunk.empty()) {
                    if (!concatenator.process_chunk(chunk)) break;
                }
                if (!has_next) break;
            }
            return concatenator.get_result_json();
        } else {
            return result->data;
        }
    }
    catch (...)
    {
        handle_exception();
    }
    return "";
}

std::string LLMService::slot_json(const json &data)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    std::string result_data = "";
    try
    {
        server_task_type task_type;
        std::string action = data.at("action");
        if (action == "save")
        {
            task_type = SERVER_TASK_TYPE_SLOT_SAVE;
        }
        else if (action == "restore")
        {
            task_type = SERVER_TASK_TYPE_SLOT_RESTORE;
        }
        else if (action == "erase")
        {
            task_type = SERVER_TASK_TYPE_SLOT_ERASE;
        }
        else
        {
            throw std::runtime_error("Invalid action" + action);
        }

        int id_slot = json_value(data, "id_slot", 0);

        server_task task(task_type);
        task.id = ctx_server->impl->queue_tasks.get_new_id();
        task.slot_action.slot_id = id_slot;

        if (action == "save" || action == "restore")
        {
            std::string filepath = data.at("filepath");
            task.slot_action.filename = filepath.substr(filepath.find_last_of("/\\") + 1);
            task.slot_action.filepath = filepath;
        }

        ctx_server->impl->queue_results.add_waiting_task_id(task.id);
        ctx_server->impl->queue_tasks.post(std::move(task));

        server_task_result_ptr result = ctx_server->impl->queue_results.recv(task.id);
        ctx_server->impl->queue_results.remove_waiting_task_id(task.id);

        json result_json = result->to_json();
        result_data = result_json.dump();
    }
    catch (...)
    {
        handle_exception();
    }
    return result_data;
}

void LLMService::cancel(int id_slot)
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    try
    {
        for (auto &slot : ctx_server->impl->slots)
        {
            if (slot.id == id_slot)
            {
                release_slot(slot);
                break;
            }
        }
    }
    catch (...)
    {
        handle_exception();
    }
}

int LLMService::embedding_size()
{
    if (setjmp(get_jump_point(true)) != 0)
        return 0;
    return ctx_server->impl->model_meta()["n_embd"];
}

//=========================== API ===========================//

void LLMService_Registry(LLMProviderRegistry *existing_instance)
{
    LLMProviderRegistry::inject_registry(existing_instance);
}

LLMService *LLMService_Construct(const char *model_path, int num_slots, int num_threads, int num_GPU_layers, bool flash_attention, int context_size, int batch_size, bool embedding_only, int lora_count, const char **lora_paths)
{
    std::vector<std::string> lora_paths_vector;
    if (lora_paths != nullptr && lora_count > 0)
    {
        for (int i = 0; i < lora_count; ++i)
        {
            lora_paths_vector.push_back(std::string(lora_paths[i]));
        }
    }
    LLMService* llmService = new LLMService(model_path, num_slots, num_threads, num_GPU_layers, flash_attention, context_size, batch_size, embedding_only, lora_paths_vector);
    if (get_status_code() != 0)
    {
        if (llmService != nullptr) delete llmService;
        return nullptr;
    }
    return llmService;
}

LLMService *LLMService_From_Command(const char *params_string_arr)
{
    LLMService* llmService;
    std::string params_string(params_string_arr);
    try
    {
        json j = json::parse(params_string);
        llmService = LLMService::from_params(j);
    }
    catch (const json::parse_error &)
    {
        llmService = LLMService::from_command(params_string);
    }

    if (get_status_code() != 0)
    {
        if (llmService != nullptr) delete llmService;
        return nullptr;
    }
    return llmService;
}

const char *LLMService_Command(LLMService *llm_service)
{
    return stringToCharArray(llm_service->get_command());
}

void LLMService_InjectErrorState(ErrorState *error_state)
{
    ErrorStateRegistry::inject_error_state(error_state);
}
