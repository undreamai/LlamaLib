#include "dynamic_loader.h"

#define DEFINE_ACCESSOR(name, ret, ...) \
    extern "C" __attribute__((visibility("default"))) name##_Fn get_##name(LLMBackend* backend) { \
        return backend->name; \
    }
LLM_FUNCTIONS(DEFINE_ACCESSOR)
#undef DEFINE_ACCESSOR
