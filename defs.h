#pragma once

#ifdef _WIN32
#ifdef UNDREAMAI_EXPORTS
#define UNDREAMAI_API __declspec(dllexport)
#else
#define UNDREAMAI_API __declspec(dllimport)
#endif
#else
#define UNDREAMAI_API
#endif