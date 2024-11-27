#include "undreamai.h"

//============================= ERROR HANDLING =============================//

void server_log_callback(const char * level, const char * function, int line, const char * message, const json & extra) {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = json{
        {"tid",       ss_tid.str()},
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
    printf("%s\n", str.c_str());
    if(logStringWrapper != nullptr) logStringWrapper->AddContent(str+"\n");
    fflush(stdout);
}

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

static void handle_terminate(){
    handle_signal_code(1);
}

#ifdef _WIN32
void set_error_handlers() {
    init_status();

    signal(SIGSEGV, handle_signal_code);
    signal(SIGFPE, handle_signal_code);

    std::set_terminate(handle_terminate);
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

    std::set_terminate(handle_terminate);
}
#endif

//============================= LLM IMPLEMENTATION =============================//

std::vector<std::string> LLM::splitArguments(const std::string& inputString) {
    std::vector<std::string> arguments;

    unsigned counter = 0;
    std::string segment;
    std::istringstream stream_input(inputString);
    while(std::getline(stream_input, segment, '"'))
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

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
EVP_PKEY* LLM::load_key(const std::string& key_str) {
    BIO *bio = BIO_new_mem_buf(key_str.data(), (int) key_str.size());
    if (!bio) return NULL;
    EVP_PKEY *key = PEM_read_bio_PrivateKey(bio, NULL, 0, NULL);
    BIO_free(bio);
    return key;
}

X509* LLM::load_cert(const std::string& cert_str) {
    BIO *bio = BIO_new_mem_buf(cert_str.data(), (int) cert_str.size());
    if (!bio) return NULL;
    X509 *cert = (cert_str[0] == '-')
                 ? PEM_read_bio_X509(bio, NULL, NULL, NULL)
                 : d2i_X509_bio(bio, NULL);
    BIO_free(bio);
    return cert;
}
#endif

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
        ctx_server.batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
            throw std::runtime_error("Invalid parameters!");
        }

        common_init();

        // enabling this will output extra debug information in the HTTP responses from the server
        // see format_final_response_oaicompat()
        const bool verbose = params.verbosity > 9;

        if (params.model_alias == "unknown") {
            params.model_alias = params.model;
        }

        llama_backend_init();
        llama_backend_has_init = true;
        llama_numa_init(params.numa);

        LOG_INFO("build info", {
            {"build",  LLAMA_BUILD_NUMBER},
            {"commit", LLAMA_COMMIT}
        });

        LOG_INFO("system info", {
            {"n_threads",       params.cpuparams.n_threads},
            {"n_threads_batch", params.cpuparams_batch.n_threads},
            {"total_threads",   std::thread::hardware_concurrency()},
            {"system_info",     llama_print_system_info()},
        });

        // Necessary similarity of prompt for slot selection
        ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

        // load the model
        if (!ctx_server.load_model(params)) {
            throw std::runtime_error("Error loading the model!");
        } else {
            ctx_server.init();
        }

        LOG_INFO("model loaded", {});

        ctx_server.queue_tasks.on_new_task(std::bind(
            &server_context::process_single_task, &ctx_server, std::placeholders::_1));
        ctx_server.queue_tasks.on_update_slots(std::bind(
            &server_context::update_slots, &ctx_server));
    } catch(...) {
        handle_exception(1);
    }
}

const json handle_post(const httplib::Request & req, httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    return json::parse(req.body);
};

void res_ok(httplib::Response & res, std::string data){
    res.set_content(data, MIMETYPE_JSON);            
    res.status = 200;
};

void handle_error(httplib::Response & res, const json error_data){
    json final_response {{"error", error_data}};
    res_ok(res, final_response.dump(-1, ' ', false, json::error_handler_t::replace));
    res.status = json_value(error_data, "code", 500);
}

void LLM::start_server(){
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INFO("Running with SSL", {{"key", params.ssl_file_key}, {"cert", params.ssl_file_cert}});
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else if (SSL_cert != "" && SSL_key != "") {
        LOG_INFO("Running with SSL", {});
        svr.reset(
            new httplib::SSLServer(load_cert(SSL_cert), load_key(SSL_key))
        );
    } else {
        LOG_INFO("Running without SSL", {});
        svr.reset(new httplib::Server());
    }
#else
    svr.reset(new httplib::Server());
#endif

    svr->set_default_headers({{"Server", "llama.cpp"}});

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, json error_data) {
        handle_error(res, error_data);
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_WARNING("got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    if (!svr->bind_to_port(params.hostname, params.port)) {
        throw std::runtime_error("couldn't bind to server socket: hostname=" + params.hostname + " port=" + std::to_string(params.port));
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // register server middlewares
    svr->set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) {
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
        return httplib::Server::HandlerResponse::Unhandled;
    });

    const auto handle_template_post = [this](const httplib::Request & req, httplib::Response & res) {
        handle_post(req, res);
        return res_ok(res, handle_template());
    };

    const auto handle_completions_post = [this, &res_error](const httplib::Request & req, httplib::Response & res) {
        json data = handle_post(req, res);
        handle_completions(data, nullptr, &res);
    };

    const auto handle_tokenize_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_tokenize(handle_post(req, res)));
    };

    const auto handle_detokenize_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_detokenize(handle_post(req, res)));
    };

    const auto handle_embeddings_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_embeddings(handle_post(req, res), &res));
    };

    const auto handle_lora_adapters_list_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_lora_adapters_list());
    };

    const auto handle_lora_adapters_apply_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_lora_adapters_apply(handle_post(req, res), &res));
    };

    const auto handle_slots_action_post = [this](const httplib::Request & req, httplib::Response & res) {
        return res_ok(res, handle_slots_action(handle_post(req, res), &res));
    };

    //
    // Router
    //

    // register API routes
    svr->Post("/completion",          handle_completions_post); // legacy
    svr->Post("/completions",         handle_completions_post);
    svr->Post("/v1/completions",      handle_completions_post);
    svr->Post("/template",            handle_template_post);
    svr->Post("/tokenize",            handle_tokenize_post);
    svr->Post("/detokenize",          handle_detokenize_post);
    svr->Post("/embedding",           handle_embeddings_post); // legacy
    svr->Post("/embeddings",          handle_embeddings_post);
    svr->Post("/v1/embeddings",       handle_embeddings_post);
    svr->Get ("/lora-adapters",       handle_lora_adapters_list_post);
    svr->Post("/lora-adapters-list",  handle_lora_adapters_list_post);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply_post);
    svr->Post("/slots",               handle_slots_action_post);

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [this] { return new httplib::ThreadPool(params.n_threads_http); };

    // run the HTTP server in a thread - see comment below
    t = std::thread([&]() {
        if (!svr->listen_after_bind()) {
            return 1;
        }

        return 0;
    });
    svr->wait_until_ready();
    
    LOG_INFO("HTTP server is listening", {{"hostname", params.hostname.c_str()}, {"port", params.port}, {"threads", params.n_threads_http}});
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
    try {
        LOG_INFO("shutting down tasks", {});
        ctx_server.queue_tasks.terminate();

        server_task_result res;
        res.stop     = true;
        res.error    = false;

        for(int id_task:ctx_server.queue_results.waiting_task_ids){
            res.id       = id_task;
            res.data     = format_error_response("shutting down task " + std::to_string(id_task), ERROR_TYPE_INVALID_REQUEST);
            ctx_server.queue_results.send(res);
        }

        if(llama_backend_has_init) llama_backend_free();
        LOG_INFO("service stopped", {});
    } catch(...) {
        handle_exception();
    }
}

bool LLM::is_running(){
    return ctx_server.queue_tasks.running;
}

void LLM::set_template(const char* chatTemplate){
    this->chatTemplate = chatTemplate;
}

void LLM::set_SSL(const char* SSL_cert, const char* SSL_key){
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    throw std::runtime_error("SSL is not supported in this build");
#endif
    this->SSL_cert = SSL_cert;
    this->SSL_key = SSL_key;
}


bool LLM::middleware_validate_api_key(const httplib::Request & req, httplib::Response & res) {
    // TODO: should we apply API key to all endpoints, including "/health" and "/models"?
    static const std::set<std::string> public_endpoints = {
        "/health",
        "/models",
        "/v1/models",
    };

    // If API key is not set, skip validation
    if (params.api_keys.empty()) {
        return true;
    }

    // If path is in public_endpoints list, skip validation
    if (public_endpoints.find(req.path) != public_endpoints.end()) {
        return true;
    }

    // Check for API key in the header
    auto auth_header = req.get_header_value("Authorization");

    std::string prefix = "Bearer ";
    if (auth_header.substr(0, prefix.size()) == prefix) {
        std::string received_api_key = auth_header.substr(prefix.size());
        if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
            return true; // API key is valid
        }
    }

    // API key is invalid or not provided
    handle_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

    LOG_WARNING("Unauthorized: Invalid API Key\n", {});

    return false;
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
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.ctx, body.at("content"), add_special, true);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.ctx, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        const json data = format_tokenizer_response(tokens_response);
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
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return data.dump();
    } catch(...) {
        handle_exception();
    }
    return "";
}

std::string LLM::handle_embeddings(json body, httplib::Response* res) {
    bool is_openai = false;

    // an input prompt can be a string or a list of tokens (integer)
    json prompt;
    if (body.count("input") != 0) {
        is_openai = true;
        prompt = body.at("input");
    } else if (body.count("content") != 0) {
        // with "content", we only support single prompt
        prompt = std::vector<std::string>{body.at("content")};
    } else {
        std::string error = "\"input\" or \"content\" must be provided";
        LOG_ERROR(error.c_str(), {});
        if(res != nullptr) handle_error(*res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
        return "";
    }

    // create and queue the task
    json responses = json::array();
    bool error = false;
    {
        std::vector<server_task> tasks = ctx_server.create_tasks_inference({{"prompt", prompt}}, SERVER_TASK_INF_TYPE_EMBEDDING);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        // get the result
        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        ctx_server.receive_cmpl_results(task_ids, [&](std::vector<server_task_result> & results) {
            for (const auto & res : results) {
                responses.push_back(res.data);
            }
        }, [&](const json & error_data) {
            handle_error(*res, error_data);
            error = true;
        });

        ctx_server.queue_results.remove_waiting_task_ids(task_ids);
    }

    if (error) {
        return "";
    }

    // write JSON response
    json root = is_openai
        ? format_embeddings_response_oaicompat(body, responses)
        : responses[0];
    return root.dump();
};

std::string LLM::handle_lora_adapters_apply(json body, httplib::Response* res) {
    int max_idx = ctx_server.loras.size();

    // clear existing value
    for (auto & lora : ctx_server.loras) {
        lora.scale = 0.0f;
    }

    // set value
    for (auto entry : body) {
        int id      = entry.at("id");
        float scale = entry.at("scale");
        if (0 <= id && id < max_idx) {
            ctx_server.loras[id].scale = scale;
        } else {
            std::string error = "invalid adapter id";
            LOG_ERROR(error.c_str(), {});
            if(res != nullptr) handle_error(*res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return "";
        }
    }

    server_task task;
    task.type = SERVER_TASK_TYPE_SET_LORA;
    const int id_task = ctx_server.queue_tasks.post(task);
    ctx_server.queue_results.add_waiting_task_id(id_task);

    server_task_result result = ctx_server.queue_results.recv(id_task);
    ctx_server.queue_results.remove_waiting_task_id(id_task);

    return result.data.dump();
};

std::string LLM::handle_lora_adapters_list(){
    json result = json::array();
    for (size_t i = 0; i < ctx_server.loras.size(); ++i) {
        auto & lora = ctx_server.loras[i];
        result.push_back({
            {"id", i},
            {"path", lora.path},
            {"scale", lora.scale},
        });
    }
    return result.dump();
}

std::string LLM::handle_completions_non_streaming(std::unordered_set<int> task_ids, httplib::Response* res) {
    std::string result_data = "";
    ctx_server.receive_cmpl_results(task_ids, [&](std::vector<server_task_result> & results) {
        if (results.size() == 1) {
            // single result
            result_data = results[0].data.dump(-1, ' ', false, json::error_handler_t::replace);
        } else {
            // multiple results (multitask)
            json arr = json::array();
            for (const auto & res : results) {
                arr.push_back(res.data);
            }
            result_data = arr.dump(-1, ' ', false, json::error_handler_t::replace);
        }
        if(res != nullptr) res_ok(*res, result_data);
    }, [&](const json & error_data) {
        LOG_ERROR("Error processing handle_completions_non_streaming request", {});
        if(res != nullptr) handle_error(*res, error_data);
    });
    return result_data;
}

class SinkException : public std::exception {};

std::string LLM::handle_completions_streaming(std::unordered_set<int> task_ids, StringWrapper* stringWrapper, httplib::DataSink* sink) {
    std::string result_data = "";

    ctx_server.receive_cmpl_results_stream(task_ids, [&](const server_task_result & result) -> bool {
        std::string str = result.data.dump(-1, ' ', false, json::error_handler_t::replace);
        str = "data: " + str + "\n\n";
        result_data += str;
        if (stringWrapper != nullptr) stringWrapper->SetContent(result_data);
        if (sink != nullptr)
        {
            if(!server_sent_event(*sink, "data", result.data)) throw SinkException();
        }
        return true;
    }, [&](const json & error_data) {
        result_data =
            "error: " +
            error_data.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n";
        if(stringWrapper != nullptr) stringWrapper->SetContent(result_data);
        if (sink != nullptr)
        {
            if(!server_sent_event(*sink, "error", error_data)) throw SinkException();
        }
    });
    return result_data;
}

std::string LLM::handle_completions(json data, StringWrapper* stringWrapper, httplib::Response* res) {
    if (setjmp(point) != 0) return "";
    clear_status();
    std::string result_data = "";
    try {
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(data, SERVER_TASK_INF_TYPE_COMPLETION);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            result_data = handle_completions_non_streaming(task_ids, res);
            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            auto on_complete = [task_ids, this] (bool) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };
            if(res == nullptr){
                result_data = handle_completions_streaming(task_ids, stringWrapper, nullptr);
                on_complete(true);
            } else {
                const auto chunked_content_provider = [task_ids, this](size_t, httplib::DataSink & sink) {
                    handle_completions_streaming(task_ids, nullptr, &sink);
                    static const std::string ev_done = "data: [DONE]\n\n";
                    if(&sink != nullptr){
                        sink.write(ev_done.data(), ev_done.size());
                        sink.done();
                    }
                    return true;
                };
                res->set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
        }
    } catch(...) {
        handle_exception();
    }
    return result_data;
}

std::string LLM::handle_slots_action(json data, httplib::Response* res) {
    if (setjmp(point) != 0) return "";
    clear_status();
    std::string result_data = "";
    try {
        server_task task;
        int id_slot = json_value(data, "id_slot", 0);
        task.data = {
            { "id_slot", id_slot},
        };

        std::string action = data["action"];
        if (action == "save" || action == "restore") {
            if (data.count("filepath") != 0) {
                std::string filepath = data.at("filepath");

                task.data["filename"] = filepath.substr(filepath.find_last_of("/\\") + 1);
                task.data["filepath"] = filepath;
            } else {
                // deprecated
                std::string filename = data.at("filename");
                if (!fs_validate_filename(filename)) {
                    LOG_ERROR(("Invalid filename: " + filename).c_str(), {});
                    if(res != nullptr) handle_error(*res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
                    return "";
                }
                task.data["filename"] = filename;
                task.data["filepath"] = params.slot_save_path + filename;
            }
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

        result_data = result.data.dump();
        if (result.error) {
            LOG_ERROR("Error processing handle_slots_action", result_data);
            if(res != nullptr) handle_error(*res, result.data);
        } else {
            if(res != nullptr) res->set_content(result_data, "application/json");
        }
    } catch(...) {
        handle_exception();
    }
    return result_data;
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
    if (llm != nullptr) delete llm;
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

const bool LLM_Started(LLM* llm) {
    return llm->is_running();
}

const void LLM_Stop(LLM* llm) {
    llm->stop_service();
}

const void LLM_SetTemplate(LLM* llm, const char* chatTemplate){
    llm->set_template(chatTemplate);
}

const void LLM_SetSSL(LLM* llm, const char* SSL_cert, const char* SSL_key){
    llm->set_SSL(SSL_cert, SSL_key);
}

const void LLM_Tokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_tokenize(json::parse(json_data)));
}

const void LLM_Detokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_detokenize(json::parse(json_data)));
}

const void LLM_Embeddings(LLM* llm, const char* json_data, StringWrapper* wrapper){
    std::string result = llm->handle_embeddings(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Lora_Weight(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_lora_adapters_apply(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Lora_List(LLM* llm, StringWrapper* wrapper) {
    std::string result = llm->handle_lora_adapters_list();
    wrapper->SetContent(result);
}

const void LLM_Completion(LLM* llm, const char* json_data, StringWrapper* wrapper){
    std::string result = llm->handle_completions(json::parse(json_data), wrapper);
    wrapper->SetContent(result);
}

const void LLM_Slot(LLM* llm, const char* json_data, StringWrapper* wrapper) {
    std::string result = llm->handle_slots_action(json::parse(json_data));
    wrapper->SetContent(result);
}

const void LLM_Cancel(LLM* llm, int id_slot) {
    llm->handle_cancel_action(id_slot);
}

const int LLM_Status(LLM* llm, StringWrapper* wrapper) {
    wrapper->SetContent(llm->get_status_message());
    return llm->get_status();
}

const int LLM_Test() {
    return 100;
}