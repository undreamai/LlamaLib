cmake_minimum_required(VERSION 3.14)
project("undreamai" C CXX)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75;86;89")

# Add subdirectories
add_subdirectory("llama.cpp")
add_subdirectory("archchecker")

# Output override
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_property(DIRECTORY llama.cpp PROPERTY CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()

# Setup CURL
if (WIN32)
    set(CURL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/curl/include")
    set(CURL_LIBRARY "${CMAKE_SOURCE_DIR}/curl/lib/libcurl_a.lib")
    set(ZLIB_LIBRARY "${CMAKE_SOURCE_DIR}/curl/lib/zlibstatic.lib")
    add_compile_definitions(CURL_STATICLIB)
endif()

find_package(CURL REQUIRED)

set(CURL_LIBRARIES ${CURL_LIBRARIES} ${ZLIB_LIBRARY})
if (WIN32)
    set(CURL_LIBRARIES ${CURL_LIBRARIES} ws2_32 Wldap32 crypt32 Normaliz)
endif()

# llama.cpp compilation flags
if (NOT WIN32)
    set(STATIC_FLAGS "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STATIC_FLAGS}")
    foreach (target common ggml-base ggml-cpu ggml-amx ggml-blas ggml-cann ggml-cuda ggml-hip ggml-kompute ggml-metal ggml-musa ggml-rpc ggml-sycl ggml-vulkan ggml llama)
        if (TARGET ${target})
            set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${STATIC_FLAGS})
            target_compile_options(${target} PRIVATE -Wno-return-type -Wno-sometimes-uninitialized)
        endif()
    endforeach()
endif()

foreach (target common ggml-cuda ggml-hip ggml-musa)
    if (TARGET ${target})
        target_compile_definitions(${target} PUBLIC GGML_USE_TINYBLAS NDEBUG)
        if(GGML_MINIMIZE_CODE_SIZE)
            target_compile_definitions(${target} PUBLIC GGML_MINIMIZE_CODE_SIZE)
        endif()
        if(GGML_NO_IQUANTS)
            target_compile_definitions(${target} PUBLIC GGML_NO_IQUANTS)
        endif()
    endif()
endforeach()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} /Zm1000")


# helper variables and functions
set(STATIC_LIBRARY undreamai)
if(DEFINED LIBRARY_SUFFIX)
  set(STATIC_LIBRARY "${STATIC_LIBRARY}_${LIBRARY_SUFFIX}")
endif()
set(DYNAMIC_LIBRARY "undreamai_lib")

if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(LIBRARY_TYPE STATIC)
else()
    set(LIBRARY_TYPE SHARED)
endif()

function(common_compile_defs NAME)
    target_compile_features(${NAME} PRIVATE cxx_std_11)
    add_compile_definitions(${NAME} PRIVATE UNDREAMAI_EXPORTS)
    target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} llama.cpp/common llama.cpp/examples/server llama.cpp llama.cpp/include llama.cpp/ggml/include archchecker ${CURL_INCLUDE_DIR})
endfunction()

function(create_executable NAME LIBRARY)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE ${LIBRARY})
    common_compile_defs(${NAME})
    install(TARGETS ${NAME} RUNTIME)
endfunction()

# build libraries
add_library(${STATIC_LIBRARY} ${LIBRARY_TYPE} LLM_service.cpp LLM.cpp LLM_client.cpp stringwrapper.cpp logging.cpp error_handling.cpp)
target_link_libraries(${STATIC_LIBRARY} PRIVATE common ${CMAKE_THREAD_LIBS_INIT} ${CURL_LIBRARIES})
common_compile_defs(${STATIC_LIBRARY})
install(TARGETS ${STATIC_LIBRARY} LIBRARY)

add_library(${DYNAMIC_LIBRARY} ${LIBRARY_TYPE} LLM_lib.cpp LLM.cpp LLM_client.cpp stringwrapper.cpp error_handling.cpp)
target_link_libraries(${DYNAMIC_LIBRARY} PRIVATE common ${CURL_LIBRARIES} FeatureDetector archchecker)
common_compile_defs(${DYNAMIC_LIBRARY})
install(TARGETS ${DYNAMIC_LIBRARY} LIBRARY)

if (BUILD_UNDREAMAI_BINARIES)
    # build executables
    create_executable(undreamai_test ${STATIC_LIBRARY})
    create_executable(undreamai_server ${STATIC_LIBRARY})
    create_executable(undreamai_dynamic ${DYNAMIC_LIBRARY})
    create_executable(undreamai_client ${STATIC_LIBRARY})

    target_link_libraries(undreamai_test PRIVATE ${DYNAMIC_LIBRARY})
    if (LLAMA_SERVER_SSL)
        find_package(OpenSSL REQUIRED)
        target_link_libraries(undreamai_server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(undreamai_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    endif()
endif()
