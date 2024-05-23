#include "undreamai.h"

//============================= ERROR HANDLING =============================//

void fail(std::string message, int code=1) {
    status = code;
    status_message = message;
}

void handle_exception(int code=-1) {
    try {
        throw;
    } catch(const std::exception& ex) {
        fail(ex.what(), code);
    } catch(...) {
        fail("Caught unknown exception", code);
    }
}


void init_status() {
    status = 0;
    status_message = "";
}

void clear_status() {
    if (status < 0) init_status();
}

static void handle_signal_code(int sig){
    fail("Severe error occured", sig);
    longjmp(point, 1);
}

#ifdef _WIN32
void set_error_handlers() {
    init_status();
    signal(SIGSEGV, handle_signal_code);
    signal(SIGFPE, handle_signal_code);
}
#else
static void handle_signal(int sig, siginfo_t *dont_care, void *dont_care_either)
{
    handle_signal_code(sig);
}

void set_error_handlers() {
    init_status();
    struct sigaction sa;

    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);

    sa.sa_flags     = SA_NODEFER;
    sa.sa_sigaction = handle_signal;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}
#endif

//============================= LLM IMPLEMENTATION =============================//

std::vector<std::string> LLM::splitArguments(const std::string& inputString) {
    // Split the input string into individual arguments
    std::vector<std::string> arguments;

    unsigned counter = 0;
    std::string segment;
    std::istringstream stream_input(inputString);
    while(std::getline(stream_input, segment, '\"'))
    {
        ++counter;
        if (counter % 2 == 0)
        {
            if (!segment.empty()) arguments.push_back(segment);
        }
        else
        {
            std::istringstream stream_segment(segment);
            while(std::getline(stream_segment, segment, ' '))
                if (!segment.empty()) arguments.push_back(segment);
        }
    }
    return arguments;
}

LLM::LLM(std::string params_string){
    std::vector<std::string> arguments = splitArguments("llm " + params_string);

    // Convert vector of strings to argc and argv
    int argc = static_cast<int>(arguments.size());
    char** argv = new char*[argc];
    for (int i = 0; i < argc; ++i) {
        argv[i] = new char[arguments[i].size() + 1];
        std::strcpy(argv[i], arguments[i].c_str());
    }
    init(argc, argv);
}

LLM::LLM(int argc, char ** argv){
    init(argc, argv);
}

void LLM::init(int argc, char ** argv){
    set_error_handlers();
    if (setjmp(point) != 0) return;
    try{
        server_params_parse(argc, argv, sparams, params);

        if (!sparams.system_prompt.empty()) {
            ctx_server.system_prompt_set(json::parse(sparams.system_prompt));
        }

        if (params.model_alias == "unknown") {
            params.model_alias = params.model;
        }

        llama_backend_init();
        llama_numa_init(params.numa);

        LOG_INFO("build info", {
            {"build",  LLAMA_BUILD_NUMBER},
            {"commit", LLAMA_COMMIT}
        });

        LOG_INFO("system info", {
            {"n_threads",       params.n_threads},
            {"n_threads_batch", params.n_threads_batch},
            {"total_threads",   std::thread::hardware_concurrency()},
            {"system_info",     llama_print_system_info()},
        });
        // load the model
        if (!ctx_server.load_model(params)) {
            throw std::runtime_error("Error loading the model!");
        } else {
            ctx_server.init();
        }

        LOG_INFO("model loaded", {});

        const auto model_meta = ctx_server.model_meta();

        // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
        if (sparams.chat_template.empty()) {
            if (!ctx_server.validate_model_chat_template()) {
                LOG_ERROR("The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses", {});
                sparams.chat_template = "chatml";
            }
        }

        // print sample chat example to make it clear which template is used
        {
            json chat;
            chat.push_back({{"role", "system"},    {"content", "You are a helpful assistant"}});
            chat.push_back({{"role", "user"},      {"content", "Hello"}});
            chat.push_back({{"role", "assistant"}, {"content", "Hi there"}});
            chat.push_back({{"role", "user"},      {"content", "How are you?"}});

            const std::string chat_example = format_chat(ctx_server.model, sparams.chat_template, chat);

            LOG_INFO("chat template", {
                {"chat_example", chat_example},
                {"built_in", sparams.chat_template.empty()},
            });
        }

        ctx_server.queue_tasks.on_new_task(std::bind(
            &server_context::process_single_task, &ctx_server, std::placeholders::_1));
        ctx_server.queue_tasks.on_finish_multitask(std::bind(
            &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
        ctx_server.queue_tasks.on_update_slots(std::bind(
            &server_context::update_slots, &ctx_server));
        ctx_server.queue_results.on_multitask_update(std::bind(
            &server_queue::update_multitask,
            &ctx_server.queue_tasks,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ));
    } catch(...) {
        handle_exception(1);
    }
}

const json handle_post(const httplib::Request & req, httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    return json::parse(req.body);
};

void handle_error(httplib::Response & res, json error_data){
    json final_response {{"error", error_data}};
    res.set_content(final_response.dump(), "application/json; charset=utf-8");
    res.status = json_value(error_data, "code", 500);
}

void LLM::start_server(){
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (sparams.ssl_key_file != "" && sparams.ssl_cert_file != "") {
        LOG_INFO("Running with SSL", {{"key", sparams.ssl_key_file}, {"cert", sparams.ssl_cert_file}});
        svr.reset(
            new httplib::SSLServer(sparams.ssl_cert_file.c_str(), sparams.ssl_key_file.c_str())
        );
    } else {
        LOG_INFO("Running without SSL", {});
        svr.reset(new httplib::Server());
    }
#else
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "llama.cpp"}});

    // CORS preflight
    svr->Options(R"(.*)", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin",      req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, json error_data) {
        handle_error(res, error_data);
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_VERBOSE("Got exception", formatted_error);
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port)) {
        throw std::runtime_error("couldn't bind to server socket: hostname=" + sparams.hostname + " port=" + std::to_string(sparams.port));
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = sparams.hostname;
    log_data["port"]     = std::to_string(sparams.port);
/*
    if (sparams.api_keys.size() == 1) {
        auto key = sparams.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (sparams.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(sparams.api_keys.size()) + " keys loaded";
    }

    state.store(SERVER_STATE_READY);


    auto middleware_validate_api_key = [this, &res_error](const httplib::Request & req, httplib::Response & res) {
        // TODO: should we apply API key to all endpoints, including "/health" and "/models"?
        static const std::set<std::string> protected_endpoints = {
            "/props",
            "/completion",
            "/completions",
            "/v1/completions",
            "/chat/completions",
            "/v1/chat/completions",
            "/infill",
            "/tokenize",
            "/detokenize",
            "/embedding",
            "/embeddings",
            "/v1/embeddings",
        };

        // If API key is not set, skip validation
        if (sparams.api_keys.empty()) {
            return true;
        }

        // If path is not in protected_endpoints list, skip validation
        if (protected_endpoints.find(req.path) == protected_endpoints.end()) {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(sparams.api_keys.begin(), sparams.api_keys.end(), received_api_key) != sparams.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        // TODO: make another middleware for CORS related logic
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WARNING("Unauthorized: Invalid API Key", {});

        return false;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key](const httplib::Request & req, httplib::Response & res) {
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    //
    // Route handlers (or controllers)
    //
*/

    const auto handle_template_post = [this](const httplib::Request & req, httplib::Response & res) {
        handle_post(req, res);
        return res.set_content(handle_template(), "application/json; charset=utf-8");
    };

    const auto handle_completions_post = [this, &res_error](const httplib::Request & req, httplib::Response & res) {
        json data = handle_post(req, res);
        handle_completions(data, nullptr, &res);
    };

    const auto handle_tokenize_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res.set_content(handle_tokenize(handle_post(req, res)), "application/json; charset=utf-8");
    };

    const auto handle_detokenize_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res.set_content(handle_detokenize(handle_post(req, res)), "application/json; charset=utf-8");
    };

    //
    // Router
    //

    // register static assets routes
    if (!sparams.public_path.empty()) {
        // Set the base directory for serving static files
        svr->set_base_dir(sparams.public_path);
    }
    // register API routes
    svr->Post("/completion",          handle_completions_post); // legacy
    svr->Post("/completions",         handle_completions_post);
    svr->Post("/v1/completions",      handle_completions_post);
    svr->Post("/template",            handle_template_post);
    svr->Post("/tokenize",            handle_tokenize_post);
    svr->Post("/detokenize",          handle_detokenize_post);

    //
    // Start the server
    //
    if (sparams.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        sparams.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(sparams.n_threads_http);
    svr->new_task_queue = [this] { return new httplib::ThreadPool(sparams.n_threads_http); };

    // run the HTTP server in a thread - see comment below
    t = std::thread([&]() {
        if (!svr->listen_after_bind()) {
            state.store(SERVER_STATE_ERROR);
            return 1;
        }

        return 0;
    });

    LOG_INFO("HTTP server listening", log_data);
}

void LLM::stop_server(){
    LOG_INFO("stopping server", {});
    if (svr.get() != nullptr) {
        svr->stop();
        t.join();
    }
}

int LLM::get_status(){
    return status;
}

std::string LLM::get_status_message(){
    return status_message;
}

void LLM::start_service(){
    LOG_INFO("starting service", {});
    ctx_server.queue_tasks.start_loop();
}

void LLM::stop_service(){
    LOG_INFO("shutting down tasks", {});
    for(int id_task:ctx_server.queue_results.waiting_task_ids)
        ctx_server.send_error(id_task, -1, "shutting down", ERROR_TYPE_INVALID_REQUEST);
    ctx_server.queue_tasks.terminate();
    llama_backend_free();
}

void LLM::set_template(const char* chatTemplate){
    this->chatTemplate = chatTemplate;
}

std::string LLM::handle_template() {
    if (setjmp(point) != 0) return "";
    clear_status();
    try {
        json data = json {
            {"template", chatTemplate}
        };
        return data.dump();
    } catch(...) {
        handle_exception();
    }
    return chatTemplate;
}

std::string LLM::handle_tokenize(json body) {
    if (setjmp(point) != 0) return "";
    clear_status();
    try {
        std::vector<llama_token> tokens;
        if (body.count("content") != 0) {
            tokens = ctx_server.tokenize(body["content"], false);
        }
        const json data = format_tokenizer_response(tokens);
        return data.dump();
    } catch(...) {
        handle_exception();
    }
    return "";
}

std::string LLM::handle_detokenize(json body) {
    if (setjmp(point) != 0) return "";
    clear_status();
    try {
        std::string content;
        if (body.count("tokens") != 0) {
            const std::vector<llama_token> tokens = body["tokens"];
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return data.dump();
    } catch(...) {
        handle_exception();
    }
    return "";
}

std::string LLM::handle_completions_non_streaming(int id_task, httplib::Response* res) {
    std::string result_data = "";
    server_task_result result = ctx_server.queue_results.recv(id_task);
    if (!result.error && result.stop) {
        result_data = result.data.dump(-1, ' ', false, json::error_handler_t::replace);
        if(res != nullptr) res->set_content(result_data, "application/json; charset=utf-8");
    } else {
        LOG_ERROR("Error processing handle_completions_non_streaming request", {});
        if(res != nullptr) handle_error(*res, result.data);
    }
    return result_data;
}

class SinkException : public std::exception {};

std::string LLM::handle_completions_streaming(int id_task, StringWrapper* stringWrapper, httplib::DataSink* sink) {
    std::string result_data = "";
    while (true) {
        server_task_result result = ctx_server.queue_results.recv(id_task);
        std::string str;
        if (!result.error) {
            str = result.data.dump(-1, ' ', false, json::error_handler_t::replace);
            str = "data: " + str + "\n\n";

            LOG_VERBOSE("data stream", {
                { "to_send", str }
            });

            if (sink != nullptr && !sink->write(str.c_str(), str.size())) {
                throw SinkException();
            }

            result_data += str;

            if (result.stop) {
                break;
            }
            stringWrapper->SetContent(result_data);
        } else {
            result_data =
                "error: " +
                result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                "\n\n";

            LOG_VERBOSE("data stream", {
                { "to_send", result_data }
            });
            LOG_ERROR("Error processing handle_completions_streaming request", {});
            if (sink != nullptr && !sink->write(result_data.c_str(), result_data.size())) {
                throw SinkException();
            }
            break;
        }
    }
    return result_data;
}

std::string LLM::handle_completions(json data, StringWrapper* stringWrapper, httplib::Response* res) {
    if (setjmp(point) != 0) return "";
    clear_status();
    std::string result_data = "";
    try {
        const int id_task = ctx_server.queue_tasks.get_new_id();

        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, false, false);

        if (!json_value(data, "stream", false)) {
            result_data = handle_completions_non_streaming(id_task, res);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
        } else {
            auto on_complete = [id_task, this] (bool) {
                ctx_server.request_cancel(id_task);
                ctx_server.queue_results.remove_waiting_task_id(id_task);
            };
            if(res == nullptr){
                result_data = handle_completions_streaming(id_task, stringWrapper, nullptr);
                on_complete(true);
            } else {
                const auto chunked_content_provider = [id_task, this](size_t, httplib::DataSink & sink) {
                    bool ok = true;
                    try {
                        handle_completions_streaming(id_task, nullptr, &sink);
                    } catch (const SinkException& e) {
                        ok = false;
                    }
                    ctx_server.queue_results.remove_waiting_task_id(id_task);
                    if(ok) sink.done();
                    return ok;
                };
                res->set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
        }
    } catch(...) {
        handle_exception();
    }
    return result_data;
}

void LLM::handle_slots_action(json data) {
    if (setjmp(point) != 0) return;
    clear_status();
    try {
        server_task task;
        int id_slot = json_value(data, "id_slot", 0);
        task.data = {
            { "id_slot", id_slot},
        };

        std::string action = data["action"];
        if (action == "save" || action == "restore") {
            std::string filename = data["filename"];
            if (!validate_file_name(filename)) {
                LOG_ERROR(("Invalid filename: " + filename).c_str(), {});
                return;
            }
            task.data["filename"] = filename;
            task.data["filepath"] = sparams.slot_save_path + filename;

            if (action == "save") {
                task.type = SERVER_TASK_TYPE_SLOT_SAVE;
            } else {
                task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
            }
        } else if (action == "erase") {
            task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        } else {
            throw std::runtime_error("Invalid action" + action);
        }

        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);
    } catch(...) {
        handle_exception();
    }
}

void LLM::handle_cancel_action(int id_slot) {
    if (setjmp(point) != 0) return;
    clear_status();
    try {
        for (auto & slot : ctx_server.slots) {
            if (slot.id == id_slot) {
                slot.release();
                break;
            }
        }
    } catch(...) {
        handle_exception();
    }
}

//============================= API IMPLEMENTATION =============================//

std::list<StringWrapper*> stringWrappers;

const void Logging(StringWrapper* wrapper)
{
    logStringWrapper = wrapper;
}

const void StopLogging()
{
    logStringWrapper = nullptr;
}

StringWrapper* StringWrapper_Construct() {
    return new StringWrapper();
}

const void StringWrapper_Delete(StringWrapper* object) {
    if(object != nullptr){
        delete object;
        object = nullptr;
    }
}

const int StringWrapper_GetStringSize(StringWrapper* object) {
    if(object == nullptr) return 0;
    return object->GetStringSize();
}

const void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize, bool clear){
    if(object == nullptr) return;
    return object->GetString(buffer, bufferSize, clear);
}

LLM* LLM_Construct(const char* params_string) {
    return new LLM(std::string(params_string));
}

const void LLM_Delete(LLM* llm) {
    delete llm;
}

const void LLM_StartServer(LLM* llm) {
    llm->start_server();
}

const void LLM_StopServer(LLM* llm) {
    llm->stop_server();
}

const void LLM_Start(LLM* llm) {
    llm->start_service();
}

const void LLM_Stop(LLM* llm) {
    llm->stop_service();
}

const void LLM_SetTemplate(LLM* llm, const char* chatTemplate){
    llm->set_template(chatTemplate);
}

const void LLM_Tokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_tokenize(json::parse(json_data)));
}

const void LLM_Detokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_detokenize(json::parse(json_data)));
}

const void LLM_Completion(LLM* llm, const char* json_data, StringWrapper* wrapper){
    std::string result = llm->handle_completions(json::parse(json_data), wrapper);
    wrapper->SetContent(result);
}

const void LLM_Slot(LLM* llm, const char* json_data) {
    llm->handle_slots_action(json::parse(json_data));
}

const void LLM_Cancel(LLM* llm, int id_slot) {
    llm->handle_cancel_action(id_slot);
}

const int LLM_Status(LLM* llm, StringWrapper* wrapper) {
    wrapper->SetContent(llm->get_status_message());
    return llm->get_status();
}

/*
int main(int argc, char ** argv) {
    LLM llm(argc, argv);
    llm.start_server();
    llm.start_service();
}

int main(int argc, char ** argv) {
    LLM llm(argc, argv);
    json data;
    // data = {
    //     {"id_slot", 0},
    //     {"action", "restore"},
    //     {"filename", "la.txt"}
    // };
    // llm.handle_slots_action(data);
    
    std::cout<<"Run prompt"<<std::endl;
    data = {
        {"prompt", "<s>[INST] A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### user: hi [/INST]### assistant:"},
        {"cache_prompt", true},
        {"id_slot", 0},
        {"stop", json::array({"[INST]", "[/INST]", "###"})}
    };
    std::cout<<llm.handle_completions(data)<<std::endl<<std::endl;
   
    data = {
        {"id_slot", 0},
        {"action", "save"},
        {"filename", "la.txt"}
    };
    llm.handle_slots_action(data);

    return 1;
}
*/
