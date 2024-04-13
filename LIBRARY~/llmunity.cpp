#include "llmunity.h"

bool LLMParser::readline(std::string & line) {
    if (!std::getline(std::cin, line)) {
        // Input stream is bad or EOF received
        line.clear();
        return false;
    }
    bool ret = false;
    if (!line.empty()) {
        char last = line.back();
        if (last == '/') { // Always return control on '/' symbol
            line.pop_back();
            return false;
        }
        if (last == '\\') { // '\\' changes the default action
            line.pop_back();
            ret = true;
        }
    }
    line += '\n';

    // By default, continue input if multiline_input is set
    return ret;
}

std::vector<std::string> LLMParser::splitArguments(const std::string& inputString) {
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
    std::vector<std::string> arguments = LLMParser::splitArguments("llm " + params_string);

    // Convert vector of strings to argc and argv
    int argc = static_cast<int>(arguments.size());
    char** argv = new char*[argc];
    for (int i = 0; i < argc; ++i) {
        argv[i] = new char[arguments[i].size() + 1];
        std::strcpy(argv[i], arguments[i].c_str());
    }
    LLM(argc, argv);
}

LLM::LLM(int argc, char ** argv){
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
    
    server_thread = std::thread(&LLM::run_server, this);
}

LLM::~LLM(){
    std::cout<<"bye"<<std::endl;
    ctx_server.queue_tasks.terminate();
    llama_backend_free();
    server_thread.join();
}

void LLM::run_server(){
    ctx_server.queue_tasks.start_loop();
}

void LLM::run_task(){
    LOG_VERBOSE("new task may arrive", {});
    while(true){
        std::unique_lock<std::mutex> lock(ctx_server.queue_tasks.mutex_tasks);
        if (ctx_server.queue_tasks.queue_tasks.empty()) {
            lock.unlock();
            break;
        }
        server_task task = ctx_server.queue_tasks.queue_tasks.front();
        ctx_server.queue_tasks.queue_tasks.erase(ctx_server.queue_tasks.queue_tasks.begin());
        lock.unlock();
        LOG_VERBOSE("callback_new_task", {{"id_task", task.id}});
        LOG_VERBOSE("callback_new_task", {{"task_type", task.type}});
        ctx_server.queue_tasks.callback_new_task(task);
    }

    LOG_VERBOSE("update_multitasks", {});

    // check if we have any finished multitasks
    auto queue_iterator = ctx_server.queue_tasks.queue_multitasks.begin();
    while (queue_iterator != ctx_server.queue_tasks.queue_multitasks.end()) {
        if (queue_iterator->subtasks_remaining.empty()) {
            // all subtasks done == multitask is done
            server_task_multi current_multitask = *queue_iterator;
            ctx_server.queue_tasks.callback_finish_multitask(current_multitask);
            // remove this multitask
            queue_iterator = ctx_server.queue_tasks.queue_multitasks.erase(queue_iterator);
        } else {
            ++queue_iterator;
        }
    }

    // all tasks in the current loop is processed, slots data is now ready
    LOG_VERBOSE("callback_update_slots", {});

    ctx_server.queue_tasks.callback_update_slots();

    LOG_VERBOSE("wait for new task", {});
    {
        std::unique_lock<std::mutex> lock(ctx_server.queue_tasks.mutex_tasks);
        if (ctx_server.queue_tasks.queue_tasks.empty()) {
            if (!ctx_server.queue_tasks.running) {
                LOG_VERBOSE("ending start_loop", {});
                return;
            }
            ctx_server.queue_tasks.condition_tasks.wait(lock, [&]{
                return (!ctx_server.queue_tasks.queue_tasks.empty() || !ctx_server.queue_tasks.running);
            });
        }
    }
}

std::string LLM::handle_tokenize(json body) {
    std::vector<llama_token> tokens;
    if (body.count("content") != 0) {
        tokens = ctx_server.tokenize(body["content"], false);
    }
    const json data = format_tokenizer_response(tokens);
    return data.dump();
}

std::string LLM::handle_detokenize(json body) {
    std::string content;
    if (body.count("tokens") != 0) {
        const std::vector<llama_token> tokens = body["tokens"];
        content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
    }

    const json data = format_detokenized_response(content);
    return data.dump();
}


std::string LLM::handle_completions(json data) {
    std::string result_data = "";
    const int id_task = ctx_server.queue_tasks.get_new_id();

    ctx_server.queue_results.add_waiting_task_id(id_task);
    ctx_server.request_completion(id_task, -1, data, false, false);
            for (int i = 0; i < (int) ctx_server.queue_results.queue_results.size(); i++) {
                if (ctx_server.queue_results.queue_results[i].id == id_task) {
                    std::cout<<"hell yeah"<<std::endl;
                }
            }

    if (!json_value(data, "stream", false)) {
        server_task_result result = ctx_server.queue_results.recv(id_task);
        if (!result.error && result.stop) {
            // res.set_content(result.data.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
            result_data =
                "data: " +
                result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                "\n\n";
            LOG_VERBOSE("data stream", {
                { "to_send", result_data }
            });
        } else {
            // res_error(res, result.data);
            LOG_ERROR("Error processing request", {});
        }

        ctx_server.queue_results.remove_waiting_task_id(id_task);
    } else {
        // const auto chunked_content_provider = [id_task, &ctx_server](size_t, httplib::DataSink & sink) {
            while (true) {
                server_task_result result = ctx_server.queue_results.recv(id_task);
                std::string str;
                if (!result.error) {
                    str =
                        "data: " +
                        result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                        "\n\n";

                    LOG_VERBOSE("data stream", {
                        { "to_send", str }
                    });

                    // if (!sink.write(str.c_str(), str.size())) {
                    //     ctx_server.queue_results.remove_waiting_task_id(id_task);
                    //     return false;
                    // }

                    if (result.stop) {
                        break;
                    }
                } else {
                    str =
                        "error: " +
                        result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                        "\n\n";

                    LOG_VERBOSE("data stream", {
                        { "to_send", str }
                    });

                    // if (!sink.write(str.c_str(), str.size())) {
                    //     ctx_server.queue_results.remove_waiting_task_id(id_task);
                    //     return false;
                    // }

                    break;
                }
                result_data += str;
            }

            // ctx_server.queue_results.remove_waiting_task_id(id_task);
            // sink.done();

            // return true;
        // };

        // auto on_complete = [id_task, &ctx_server] (bool) {
            // cancel
            ctx_server.request_cancel(id_task);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
        // };

        // res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
    }
    return result_data;
}

int main(int argc, char ** argv) {
    LLM llm(argc, argv);
    std::cout<<"Run prompt"<<std::endl;
    json data = {
        {"prompt", "<s>[INST] A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### user: hi [/INST]### assistant:"},
        {"stop", json::array({"[INST]", "[/INST]", "###"})}
    };
    std::cout<<llm.handle_completions(data);
    return 1;
}
