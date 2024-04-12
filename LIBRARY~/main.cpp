#include "llmunity.cpp"

int main(int argc, char ** argv) {
    gpt_params params;
    if (argc <= 1){
        std::string params_string =  R"(-i -m /home/benuix/models/phi-2.Q4_K_M.gguf -c 2048 --keep 256 --repeat_penalty 1.1 --prompt "Transcript of a dialog, where the User interacts with an Assistant named Lucy. Lucy is a friendly dinosaur." -r "User:" -s 1234)";
        params = LLMParser::string_to_gpt_params(params_string);
    }else {
        if (!gpt_params_parse(argc, argv, params)) {
            return 1;
        }
    }
    // g_params = &params;

    // if (!gpt_params_parse(argc, argv, params)) {
    //     return 1;
    // }
    // llama_sampling_params & sparams = params.sparams;

// #ifndef LOG_DISABLE_LOGS
//     log_set_target(log_filename_generator("main", "log"));
//     LOG_TEE("Log start\n");
//     log_dump_cmdline(argc, argv);
//     llama_log_set(llama_log_callback_logTee, nullptr);
// #endif // LOG_DISABLE_LOGS

    // TODO: Dump params ?
    //LOG("Params perplexity: %s\n", LOG_TOSTR(params.perplexity));

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    // console::init(params.simple_io, params.use_color);
    // atexit([]() { console::cleanup(); });

    // if (!check_params(params)) return 0;

    LLM llm(params);
    llm.run();
}