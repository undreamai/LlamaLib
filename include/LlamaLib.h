/// @file LlamaLib.h
/// @brief Main include file for the LLama library
/// @ingroup llm
/// @details This convenience header includes all major components of the LLM library:
/// service implementation, runtime loading, and client functionality

#include "LLM_service.h" ///< LLM service implementation
#include "LLM_runtime.h" ///< Dynamic library loading and runtime management
#include "LLM_client.h"  ///< Client implementation for local/remote LLM access