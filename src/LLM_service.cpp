#include "LLM_service.h"

#include "common.h"
#include "llama-chat.h"
#include "log.h"
#ifndef SERVER_H
#define SERVER_H
#include "server.cpp"
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

EVP_PKEY *load_key(const std::string &key_str)
{
    BIO *bio = BIO_new_mem_buf(key_str.data(), (int)key_str.size());
    if (!bio)
        return NULL;
    EVP_PKEY *key = PEM_read_bio_PrivateKey(bio, NULL, 0, NULL);
    BIO_free(bio);
    return key;
}

X509 *load_cert(const std::string &cert_str)
{
    BIO *bio = BIO_new_mem_buf(cert_str.data(), (int)cert_str.size());
    if (!bio)
        return NULL;
    X509 *cert = (cert_str[0] == '-')
                     ? PEM_read_bio_X509(bio, NULL, NULL, NULL)
                     : d2i_X509_bio(bio, NULL);
    BIO_free(bio);
    return cert;
}

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
        LLMProviderRegistry &registry = LLMProviderRegistry::instance();
        registry.register_instance(this);
        debug(registry.get_debug_level());
        logging_callback(registry.get_log_callback());

        ctx_server = new server_context();
        ctx_server->batch = {0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        params = new common_params();
        params->port = 0;
        if (!common_params_parse(argc, argv, *params, LLAMA_EXAMPLE_SERVER))
        {
            throw std::runtime_error("Invalid parameters!");
        }

        common_init();

        llama_backend_init();
        llama_backend_has_init = true;
        llama_numa_init(params->numa);

        LLAMALIB_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params->cpuparams.n_threads, params->cpuparams_batch.n_threads, std::thread::hardware_concurrency());

        // Necessary similarity of prompt for slot selection
        ctx_server->slot_prompt_similarity = params->slot_prompt_similarity;

        // load the model
        if (!ctx_server->load_model(*params))
        {
            throw std::runtime_error("Error loading the model!");
        }
        else
        {
            ctx_server->init();
        }
        LLAMALIB_INF("model loaded\n");

        params->chat_template = detect_chat_template();
        LLAMALIB_INF("chat_template: %s\n", params->chat_template.c_str());

        ctx_server->queue_tasks.on_new_task([this](server_task && task)
                                            { this->ctx_server->process_single_task(std::move(task)); });
        ctx_server->queue_tasks.on_update_slots([this]()
                                                { this->ctx_server->update_slots(); });
    }
    catch (...)
    {
        handle_exception(1);
    }
}

const std::string LLMService::detect_chat_template()
{
    const char *chat_template_jinja = common_chat_templates_source(ctx_server->chat_templates.get());
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
    res.set_content(data, MIMETYPE_JSON);
    res.status = 200;
};

void handle_error(httplib::Response &res, const json error_data)
{
    json final_response{{"error", error_data}};
    res.set_content(final_response.dump(), MIMETYPE_JSON);
    res.status = 500;
}

void release_slot(server_slot &slot)
{
    if (slot.task_type == SERVER_TASK_TYPE_COMPLETION)
    {
        slot.i_batch = -1;
        slot.params.n_predict = 0;
        slot.stop = STOP_TYPE_LIMIT;
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
    if (ctx_server->slots.size() == 0)
        return -1;
    return next_available_slot++ % ctx_server->slots.size();
}

void LLMService::start_server(const std::string &host, int port, const std::string &API_key)
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    params->hostname = host.empty() ? "0.0.0.0" : host;
    if (port >= 0)
        params->port = port;
    params->api_keys.clear();
    if (!API_key.empty())
        params->api_keys.push_back(API_key);

    std::lock_guard<std::mutex> lock(start_stop_mutex);
    if (params->ssl_file_key != "" && params->ssl_file_cert != "")
    {
        LLAMALIB_INF("Running with SSL: key = %s, cert = %s\n", params->ssl_file_key.c_str(), params->ssl_file_cert.c_str());
        svr.reset(
            new httplib::SSLServer(params->ssl_file_cert.c_str(), params->ssl_file_key.c_str()));
    }
    else if (SSL_cert != "" && SSL_key != "")
    {
        LLAMALIB_INF("Running with SSL\n");
        svr.reset(
            new httplib::SSLServer(load_cert(SSL_cert), load_key(SSL_key)));
    }
    else
    {
        LLAMALIB_INF("Running without SSL\n");
        svr.reset(new httplib::Server());
    }

    svr->set_default_headers({{"Server", "llama.cpp"}});

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response &res, json error_data)
    {
        handle_error(res, error_data);
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response &res, const std::exception_ptr &ep)
                               {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        try {
            json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
            LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
            res_error(res, formatted_error);
        } catch (const std::exception & e) {
            LOG_ERR("got exception: %s\n", e.what());
        } });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response &res)
                           {
                               if (res.status == 404)
                               {
                                   res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
                               }
                               // for other error codes, we skip processing here because it's already done by res_error()
                           });

    // set timeouts and change hostname and port
    svr->set_read_timeout(params->timeout_read);
    svr->set_write_timeout(params->timeout_write);

    bool was_bound = false;
    bool is_sock = false;
    if (string_ends_with(std::string(params->hostname), ".sock"))
    {
        is_sock = true;
        LOG_INF("%s: setting address family to AF_UNIX\n", __func__);
        svr->set_address_family(AF_UNIX);
        // bind_to_port requires a second arg, any value other than 0 should
        // simply get ignored
        was_bound = svr->bind_to_port(params->hostname, 8080);
    }
    else
    {
        LOG_INF("%s: binding port with default address family\n", __func__);
        // bind HTTP listen port
        if (params->port == 0)
        {
            int bound_port = svr->bind_to_any_port(params->hostname);
            if ((was_bound = (bound_port >= 0)))
            {
                params->port = bound_port;
            }
        }
        else
        {
            was_bound = svr->bind_to_port(params->hostname, params->port);
        }
    }

    if (!was_bound)
    {
        throw std::runtime_error("couldn't bind to server socket: hostname=" + params->hostname + " port=" + std::to_string(params->port));
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params->hostname;
    log_data["port"] = std::to_string(params->port);

    if (params->api_keys.size() == 1)
    {
        auto key = params->api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    }
    else if (params->api_keys.size() > 1)
    {
        log_data["api_key"] = "api_key: " + std::to_string(params->api_keys.size()) + " keys loaded";
    }

    // register server middlewares
    svr->set_pre_routing_handler([this](const httplib::Request &req, httplib::Response &res)
                                 {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled; });

    const auto completion_post = [this, &res_error](const httplib::Request &req, httplib::Response &res)
    {
        json data = handle_post(req, res);
        completion_json(data, nullptr, true, &res, req.is_connection_closed, OAICOMPAT_TYPE_NONE);
    };

    const auto chat_completion_post = [this, &res_error](const httplib::Request &req, httplib::Response &res)
    {
        json body = handle_post(req, res);
        json data = oaicompat_completion_params_parse(body);
        LOG_DBG("formatted prompt: %s\n", data.dump().c_str());
        completion_json(data, nullptr, true, &res, req.is_connection_closed, OAICOMPAT_TYPE_CHAT);
    };

    const auto get_template_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return res_ok(res, get_template());
    };

    const auto apply_template_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return res_ok(res, apply_template_json(handle_post(req, res)));
    };

    const auto tokenize_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return res_ok(res, tokenize_json(handle_post(req, res)));
    };

    const auto detokenize_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return res_ok(res, detokenize_json(handle_post(req, res)));
    };

    const auto embeddings_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return embeddings_json(handle_post(req, res), &res, req.is_connection_closed);
    };

    const auto lora_list_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return res_ok(res, lora_list_json());
    };

    const auto lora_weight_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return lora_weight_json(handle_post(req, res), &res);
    };

    const auto slots_post = [this](const httplib::Request &req, httplib::Response &res)
    {
        return slot_json(handle_post(req, res), &res);
    };

    //
    // Router
    //

    // register API routes
    svr->Post(params->api_prefix + "/completion", completion_post); // legacy
    svr->Post(params->api_prefix + "/completions", completion_post);
    svr->Post(params->api_prefix + "/chat/completions", chat_completion_post);
    svr->Post(params->api_prefix + "/v1/chat/completions", chat_completion_post);
    svr->Post(params->api_prefix + "/tokenize", tokenize_post);
    svr->Post(params->api_prefix + "/detokenize", detokenize_post);
    svr->Post(params->api_prefix + "/apply-template", apply_template_post);
    svr->Post(params->api_prefix + "/embedding", embeddings_post); // legacy
    svr->Post(params->api_prefix + "/embeddings", embeddings_post);
    svr->Post(params->api_prefix + "/get-template", get_template_post);
    // svr->Get ("/lora-adapters",       lora_list_post);
    // svr->Post(params->api_prefix + "/lora-adapters-list",  lora_list_post);
    // svr->Post(params->api_prefix + "/lora-adapters",       lora_weight_post);
    // svr->Post(params->api_prefix + "/slots",               slots_post);

    //
    // Start the server
    //
    if (params->n_threads_http < 1)
    {
        // +2 threads for monitoring endpoints
        params->n_threads_http = std::max(params->n_parallel + 2, (int32_t)std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] = std::to_string(params->n_threads_http);
    svr->new_task_queue = [this]
    { return new httplib::ThreadPool(params->n_threads_http); };

    // run the HTTP server in a thread - see comment below
    server_thread = std::thread([&]()
                                {
        if (!svr->listen_after_bind()) {
            return 1;
        }

        return 0; });
    svr->wait_until_ready();

    LLAMALIB_INF("%s: server is listening on %s - starting the main loop\n", __func__,
                 is_sock ? string_format("unix://%s", params->hostname.c_str()).c_str() :
                 string_format("http://%s:%d", params->hostname.c_str(), params->port).c_str());
}

void LLMService::stop_server()
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    std::lock_guard<std::mutex> lock(start_stop_mutex);
    LLAMALIB_INF("stopping server\n");
    if (svr.get() != nullptr)
    {
        svr->stop();
        if (server_thread.joinable())
        {
            server_thread.join();
        }
    }
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
        ctx_server->queue_tasks.start_loop();
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
        for (server_slot &slot : ctx_server->slots)
            release_slot(slot);

        // wait until tasks have completed
        if((!ctx_server->queue_tasks.queue_tasks.empty()))
        {
            LLAMALIB_INF("Wait until tasks have finished\n");
            int grace = 10;
            while (!ctx_server->queue_tasks.queue_tasks.empty() && grace-- > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            LLAMALIB_INF("Tasks have finished\n");
        }

        ctx_server->queue_tasks.terminate();
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
    return ctx_server != nullptr && ctx_server->queue_tasks.running;
}

void LLMService::set_SSL(const std::string &SSL_cert_str, const std::string &SSL_key_str)
{
    SSL_cert = SSL_cert_str;
    SSL_key = SSL_key_str;
}

bool LLMService::middleware_validate_api_key(const httplib::Request &req, httplib::Response &res)
{
    // TODO: should we apply API key to all endpoints, including "/health" and "/models"?
    static const std::set<std::string> public_endpoints = {
        "/health",
        "/models",
        "/v1/models",
    };

    // If API key is not set, skip validation
    if (params->api_keys.empty())
    {
        return true;
    }

    // If path is in public_endpoints list, skip validation
    if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/")
    {
        return true;
    }

    // Check for API key in the header
    auto auth_header = req.get_header_value("Authorization");

    std::string prefix = "Bearer ";
    if (auth_header.substr(0, prefix.size()) == prefix)
    {
        std::string received_api_key = auth_header.substr(prefix.size());
        if (std::find(params->api_keys.begin(), params->api_keys.end(), received_api_key) != params->api_keys.end())
        {
            return true; // API key is valid
        }
    }

    // API key is invalid or not provided
    handle_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

    LOG_WRN("Unauthorized: Invalid API Key\n");

    return false;
}

std::string LLMService::get_template()
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    return params->chat_template;
}

void LLMService::set_template(std::string chat_template)
{
    if (setjmp(get_jump_point(true)) != 0)
        return;
    ctx_server->chat_templates = common_chat_templates_init(ctx_server->model, chat_template);
    params->chat_template = detect_chat_template();
    ctx_server->oai_parser_opt = {
        /* use_jinja             */ params->use_jinja,
        /* prefill_assistant     */ params->prefill_assistant,
        /* reasoning_format      */ params->reasoning_format,
        /* chat_template_kwargs  */ params->default_template_kwargs,
        /* common_chat_templates */ ctx_server->chat_templates.get(),
        /* allow_image           */ false,
        /* allow_audio           */ false,
        /* enable_thinking       */ params->reasoning_budget != 0,
    };
}

std::string LLMService::apply_template(const json &messages)
{
    return parse_apply_template_json(json::parse(apply_template_json(build_apply_template_json(messages))));
}

std::string LLMService::apply_template_json(const json &body)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    std::vector<raw_buffer> files; // dummy, unused
    json copy = body;
    json data = oaicompat_chat_params_parse(
        copy,
        ctx_server->oai_parser_opt,
        files);

    std::string prompt = std::move(data.at("prompt"));
    if (!reasoning_enabled)
    {
        const std::string src = std::string(common_chat_templates_source(ctx_server->oai_parser_opt.tmpls));
        if (
            (src.find("<｜tool▁calls▁begin｜>") != std::string::npos) ||
            (src.find("<tool_call>") != std::string::npos) ||
            (src.find("elif thinking") != std::string::npos && src.find("<|tool_call|>") != std::string::npos)
        ) {
            prompt += "<think>\n</think>\n";
        } else if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos) {
            prompt += "<|START_THINKING|>\n<|END_THINKING|>\n";
        } else if (src.find("<|channel|>") != std::string::npos) {
            prompt += "<|channel|>analysis<|message|>\n<|start|>assistant<|channel|>final<|message|>\n";
        }
    }
    return safe_json_to_str({{"prompt", prompt}});
}

std::vector<int> LLMService::tokenize(const std::string &input)
{
    return parse_tokenize_json(json::parse(tokenize_json(build_tokenize_json(input))));
}

std::string LLMService::tokenize_json(const json &body)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    try
    {
        json tokens_response = json::array();
        if (body.count("content") != 0)
        {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server->vocab, body.at("content"), add_special, parse_special);

            if (with_pieces)
            {
                for (const auto &token : tokens)
                {
                    std::string piece = common_token_to_piece(ctx_server->ctx, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece))
                    {
                        piece_json = piece;
                    }
                    else
                    {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece)
                        {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({{"id", token},
                                               {"piece", piece_json}});
                }
            }
            else
            {
                tokens_response = tokens;
            }
        }

        const json data = format_tokenizer_response(tokens_response);
        return data.dump();
    }
    catch (...)
    {
        handle_exception();
    }
    return "";
}

std::string LLMService::detokenize(const std::vector<int32_t> &tokens)
{
    return parse_detokenize_json(json::parse(detokenize_json(build_detokenize_json(tokens))));
}

std::string LLMService::detokenize_json(const json &body)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    try
    {
        std::string content;
        if (body.count("tokens") != 0)
        {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return data.dump();
    }
    catch (...)
    {
        handle_exception();
    }
    return "";
}

std::vector<float> LLMService::embeddings(const std::string &query)
{
    return parse_embeddings_json(json::parse(embeddings_json(build_embeddings_json(query))));
}

std::string LLMService::embeddings_json(
    const json &body,
    httplib::Response *res,
    std::function<bool()> is_connection_closed)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    oaicompat_type oaicompat = OAICOMPAT_TYPE_EMBEDDING;
    // an input prompt can be a string or a list of tokens (integer)
    json prompt;
    if (body.count("input") != 0)
    {
        prompt = body.at("input");
    }
    else if (body.contains("content"))
    {
        prompt = body.at("content");
    }
    else
    {
        std::string error = "\"input\" or \"content\" must be provided";
        LOG_ERR("%s\n", error.c_str());
        if (res != nullptr)
            handle_error(*res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
        return "";
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0)
    {
        const std::string &format = body.at("encoding_format");
        if (format == "base64")
        {
            use_base64 = true;
        }
        else if (format != "float")
        {
            if (res != nullptr)
                handle_error(*res, format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return "";
        }
    }

    std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);
    for (const auto &tokens : tokenized_prompts)
    {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty())
        {
            if (res != nullptr)
                handle_error(*res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return "";
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (llama_pooling_type(ctx_server->ctx) == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", llama_pooling_type(ctx_server->ctx));
        }
    }

    // create and queue the task
    json responses = json::array();
    bool error = false;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++)
        {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;
            task.prompt_tokens = server_tokens(tokenized_prompts[i], ctx_server->mctx != nullptr);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }

        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(std::move(tasks));

        // get the result
        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);
        ctx_server->receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> &results)
                                          {
            for (auto & res : results) {
                server_task_result_embd* res_embd = dynamic_cast<server_task_result_embd*>(res.get());
                GGML_ASSERT(res_embd != nullptr);
                responses.push_back(res->to_json());
            } }, [&](const json &error_data)
                                          {
            if(res != nullptr) handle_error(*res, error_data);
            error = true; }, is_connection_closed);

        ctx_server->queue_results.remove_waiting_task_ids(task_ids);
    }

    if (error)
    {
        return "";
    }

    // write JSON response
    json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
                    ? format_embeddings_response_oaicompat(body, responses, use_base64)
                    : json(responses);

    // take the pooled data
    std::string result = safe_json_to_str(root["data"][0]);
    if (res != nullptr)
        res_ok(*res, result);
    return result;
};

bool LLMService::lora_weight(const std::vector<LoraIdScale> &loras)
{
    return parse_lora_weight_json(json::parse(lora_weight_json(build_lora_weight_json(loras))));
}

std::string LLMService::lora_weight_json(const json &body, httplib::Response *res)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    if (!body.is_array())
    {
        if (res != nullptr)
            handle_error(*res, format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
        return "";
    }

    std::vector<common_adapter_lora_info> lora(ctx_server->params_base.lora_adapters);
    int max_idx = lora.size();

    // clear existing value
    for (auto &entry : lora)
    {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto &entry : body)
    {
        int id = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx)
        {
            lora[id].scale = scale;
        }
        else
        {
            std::string error = "invalid adapter id";
            LOG_ERR("%s\n", error.c_str());
            if (res != nullptr)
                handle_error(*res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return "";
        }
    }

    server_task task(SERVER_TASK_TYPE_SET_LORA);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.set_lora = lora;
    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(std::move(task));

    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    json result_data = result->to_json();
    if (res != nullptr)
    {
        if (result->is_error())
        {
            handle_error(*res, result_data);
        }
        else
        {
            res_ok(*res, result_data);
        }
    }

    return safe_json_to_str(result_data);
};

std::vector<LoraIdScalePath> LLMService::lora_list()
{
    return parse_lora_list_json(json::parse(lora_list_json()));
}

std::string LLMService::lora_list_json()
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    json result = json::array();
    const auto &loras = ctx_server->params_base.lora_adapters;
    for (size_t i = 0; i < loras.size(); ++i)
    {
        auto &lora = loras[i];
        result.push_back({
            {"id", i},
            {"path", lora.path},
            {"scale", lora.scale},
        });
    }
    return result.dump();
}

class SinkException : public std::exception
{
};

static void server_sent_event_with_stringswrapper(
    CharArrayFn callback,
    bool callbackWithJSON,
    httplib::DataSink *sink,
    const char *event,
    const json &data,
    std::string &concat_string,
    std::vector<int> &concat_tokens,
    bool return_tokens)
{
    concat_string += json_value(data, "content", std::string(""));
    if (return_tokens)
    {
        for (const auto &tok : json_value(data, "tokens", std::vector<int>()))
        {
            concat_tokens.push_back(tok);
        }
    }

    if (sink != nullptr)
    {
        // in remote mode do not concat, it will be done on the receiving end to save data
        const std::string sink_str = std::string(event) + ": " + safe_json_to_str(data) + "\n\n";
        if (!sink->write(sink_str.c_str(), sink_str.size()))
            throw SinkException();
    }
    else
    {
        // in local mode, concat and call callback
        if (callback)
        {
            if (callbackWithJSON)
            {
                json data_concat = data;
                data_concat["content"] = concat_string;
                data_concat["tokens"] = concat_tokens;
                callback(safe_json_to_str(data_concat).c_str());
            }
            else
            {
                callback(concat_string.c_str());
            }
        }

    }
}

std::string LLMService::completion_streaming(
    std::unordered_set<int> task_ids,
    CharArrayFn callback,
    bool callbackWithJSON,
    bool return_tokens,
    httplib::DataSink *sink,
    std::function<bool()> is_connection_closed)
{
    std::string concat_string = "";
    std::vector<int> concat_tokens;
    json result_data;

    ctx_server->receive_cmpl_results_stream(task_ids, [&](server_task_result_ptr &result) -> bool {
        json res_json = result->to_json();
        if (res_json.is_array()) {
            for (const auto & res : res_json) {
                server_sent_event_with_stringswrapper(callback, callbackWithJSON, sink, "data", res, concat_string, concat_tokens, return_tokens);
                result_data = res;
            }
        } else {
            server_sent_event_with_stringswrapper(callback, callbackWithJSON, sink, "data", res_json, concat_string, concat_tokens, return_tokens);
            result_data = res_json;
        }
        return true; }, [&](const json &error_data) {
            server_sent_event_with_stringswrapper(callback, callbackWithJSON, sink, "error", error_data, concat_string, concat_tokens, return_tokens);
        }, is_connection_closed
    );
    if (sink != nullptr)
        sink->done();
    result_data["content"] = concat_string;
    result_data["tokens"] = concat_tokens;
    return safe_json_to_str(result_data);
}

std::string LLMService::completion_json(const json &data, CharArrayFn callback, bool callbackWithJSON)
{
    return completion_json(data, callback, callbackWithJSON, nullptr);
}

std::string LLMService::completion_json(
    const json &data,
    CharArrayFn callback,
    bool callbackWithJSON,
    httplib::Response *res,
    std::function<bool()> is_connection_closed,
    int oaicompat_int)
{
    if (setjmp(get_jump_point(true)) != 0)
        return "";
    std::string result_data = "";
    try
    {
        server_task_type type = SERVER_TASK_TYPE_COMPLETION;
        oaicompat_type oaicompat = static_cast<oaicompat_type>(oaicompat_int);
        bool stream = json_value(data, "stream", callback != nullptr);
        bool return_tokens = json_value(data, "return_tokens", false);

        // GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server->params_base.embedding)
        {
            if (res != nullptr)
                handle_error(*res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return "";
        }

        auto completion_id = gen_chatcmplid();
        std::unordered_set<int> task_ids;
        std::vector<server_task> tasks;
        try
        {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++)
            {
                server_task task = server_task(type);

                task.id = ctx_server->queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens = server_tokens(tokenized_prompts[i], ctx_server->mctx != nullptr);
                task.params = server_task::params_from_json_cmpl(
                    ctx_server->ctx,
                    ctx_server->params_base,
                    data);
                task.params.stream = stream;
                task.id_selected_slot = (res == nullptr) ? json_value(data, "id_slot", -1) : -1;

                // OAI-compat
                task.params.oaicompat = oaicompat;
                task.params.oaicompat_cmpl_id = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(std::move(task));
            }

            task_ids = server_task::get_list_id(tasks);
            ctx_server->queue_results.add_waiting_tasks(tasks);
            ctx_server->queue_tasks.post(std::move(tasks));
        }
        catch (const std::exception &e)
        {
            if (res != nullptr)
                handle_error(*res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return "";
        }

        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(std::move(tasks));

        if (!stream)
        {
            ctx_server->receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> &results)
                                              {
                json result_json;
                if (results.size() == 1) {
                    // single result
                    result_json = results[0]->to_json();
                } else {
                    // multiple results (multitask)
                    result_json = json::array();
                    for (auto & res : results) {
                        result_json.push_back(res->to_json());
                    }
                }
                result_data = safe_json_to_str(result_json);
                if (res != nullptr) res_ok(*res, result_data);
                if (callback)
                {
                    if (callbackWithJSON) callback(result_data.c_str());
                    else callback(json_value(result_data, "content", std::string("")).c_str());
                } }, [&](const json &error_data)
                                              {
                if(res != nullptr) handle_error(*res, error_data); }, is_connection_closed);

            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        }
        else
        {
            auto on_complete = [task_ids, this](bool)
            {
                ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            };
            if (res == nullptr)
            {
                result_data = completion_streaming(task_ids, callback, callbackWithJSON, return_tokens, nullptr);
                on_complete(true);
            }
            else
            {
                const auto chunked_content_provider = [task_ids, this, oaicompat, callbackWithJSON, return_tokens](size_t, httplib::DataSink &sink)
                {
                    bool ok = true;
                    try
                    {
                        completion_streaming(task_ids, nullptr, callbackWithJSON, return_tokens, &sink, [&sink]()
                                             { return !sink.is_writable(); });
                        if (oaicompat != OAICOMPAT_TYPE_NONE)
                        {
                            static const std::string ev_done = "data: [DONE]\n\n";
                            sink.write(ev_done.data(), ev_done.size());
                        }
                    }
                    catch (const SinkException &e)
                    {
                        ok = false;
                    }
                    // ctx_server->queue_results.remove_waiting_task_ids(task_ids);
                    try
                    {
                        sink.done();
                    }
                    catch (...)
                    {
                    }
                    return ok;
                };
                res->set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
        }
    }
    catch (...)
    {
        handle_exception();
    }
    return result_data;
}

std::string LLMService::slot(int id_slot, const std::string &action, const std::string &filepath)
{
    return parse_slot_json(json::parse(slot_json(build_slot_json(id_slot, action, filepath))));
}

std::string LLMService::slot_json(
    const json &data,
    httplib::Response *res)
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
        task.id = ctx_server->queue_tasks.get_new_id();
        task.slot_action.slot_id = id_slot;

        if (action == "save" || action == "restore")
        {
            std::string filepath = data.at("filepath");
            task.slot_action.filename = filepath.substr(filepath.find_last_of("/\\") + 1);
            task.slot_action.filepath = filepath;
        }

        ctx_server->queue_results.add_waiting_task_id(task.id);
        ctx_server->queue_tasks.post(std::move(task));

        server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
        ctx_server->queue_results.remove_waiting_task_id(task.id);

        json result_json = result->to_json();
        result_data = result_json.dump();
        if (result->is_error())
        {
            LOG_ERR("Error processing slots: %s\n", result_data.c_str());
            if (res != nullptr)
                handle_error(*res, result_json);
        }
        else
        {
            if (res != nullptr)
                res_ok(*res, result_json);
        }
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
        for (auto &slot : ctx_server->slots)
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
    return ctx_server->model_meta()["n_embd"];
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
    return new LLMService(model_path, num_slots, num_threads, num_GPU_layers, flash_attention, context_size, batch_size, embedding_only, lora_paths_vector);
}

LLMService *LLMService_From_Command(const char *params_string_arr)
{
    std::string params_string(params_string_arr);
    try
    {
        json j = json::parse(params_string);
        return LLMService::from_params(j);
    }
    catch (const json::parse_error &)
    {
        return LLMService::from_command(params_string);
    }
}

void LLMService_InjectErrorState(ErrorState *error_state)
{
    ErrorStateRegistry::inject_error_state(error_state);
}
