#include "archchecker.h"

const bool has_avx() {
    return ggml_cpu_has_avx();
}

const bool has_avx2() {
    return ggml_cpu_has_avx2();
}

const bool has_avx512() {
    return ggml_cpu_has_avx512();
}
