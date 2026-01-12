#pragma once
#include <iostream>

#ifdef _WIN32
#ifdef UNDREAMAI_EXPORTS
#define ARCHCHECKER_API __declspec(dllexport)
#else
#define ARCHCHECKER_API __declspec(dllimport)
#endif
#else
#define ARCHCHECKER_API
#endif

extern "C"
{
    ARCHCHECKER_API const bool has_avx();
    ARCHCHECKER_API const bool has_avx2();
    ARCHCHECKER_API const bool has_avx512();
};
