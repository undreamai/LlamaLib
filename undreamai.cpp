#include "undreamai.h"

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
    ctx_server.queue_tasks.terminate();
    llama_backend_free();
    server_thread.join();
}

void LLM::run_server(){
    ctx_server.queue_tasks.start_loop();
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


std::string LLM::handle_completions(json data, StringWrapperCallback* streamCallback) {
    std::string result_data = "";
    const int id_task = ctx_server.queue_tasks.get_new_id();

    ctx_server.queue_results.add_waiting_task_id(id_task);
    ctx_server.request_completion(id_task, -1, data, false, false);

    if (!json_value(data, "stream", false)) {
        server_task_result result = ctx_server.queue_results.recv(id_task);
        if (!result.error && result.stop) {
            result_data =
                "data: " +
                result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                "\n\n";
            LOG_VERBOSE("data stream", {
                { "to_send", result_data }
            });
        } else {
            LOG_ERROR("Error processing request", {});
        }

        ctx_server.queue_results.remove_waiting_task_id(id_task);
    } else {
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

                    break;
                }
                result_data += str;
                if(streamCallback != nullptr) streamCallback->Call(result_data);
            }

            ctx_server.request_cancel(id_task);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
    }
    return result_data;
}

void LLM::handle_slots_action(json data) {
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
}

StringWrapper::StringWrapper(){}

void StringWrapper::SetContent(std::string input){
    if (content != nullptr) {
        delete[] content;
    }
    content = new char[input.length() + 1];
    strcpy(content, input.c_str());
}

int StringWrapper::GetStringSize(){
    return strlen(content) + 1;
}

void StringWrapper::GetString(char* buffer, int bufferSize){
    strncpy(buffer, content, bufferSize);
    buffer[bufferSize - 1] = '\0';
}

StringWrapper* StringWrapper_Construct() {
    return new StringWrapper();
}

void StringWrapper_Delete(StringWrapper* object) {
    delete object;
}

int StringWrapper_GetStringSize(StringWrapper* object) {
    return object->GetStringSize();
}

void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize){
    return object->GetString(buffer, bufferSize);
}

LLM* LLM_Construct(const char* params_string) {
    return new LLM(std::string(params_string));
}

void LLM_Delete(LLM* llm) {
    delete llm;
}

const void LLM_Tokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_tokenize(json::parse(json_data)));
}

const void LLM_Detokenize(LLM* llm, const char* json_data, StringWrapper* wrapper){
    wrapper->SetContent(llm->handle_detokenize(json::parse(json_data)));
}

void LLM_Completion(LLM* llm, const char* json_data, StringWrapper* wrapper, void* streamCallbackPointer){
    StringWrapperCallback* callback = nullptr;
    if (streamCallbackPointer != nullptr){
        CompletionCallback streamCallback = reinterpret_cast<CompletionCallback>(streamCallbackPointer);
        callback = new StringWrapperCallback(wrapper, streamCallback);
    }
    wrapper->SetContent(llm->handle_completions(json::parse(json_data), callback));
}

const void LLM_Slot(LLM* llm, const char* json_data) {
    llm->handle_slots_action(json::parse(json_data));
}

/*
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
