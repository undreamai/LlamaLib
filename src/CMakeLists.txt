if(IOS_VISIONOS_HTTP)
    set(IOS_SSL_FILES ios_http_transport.mm)
    set_source_files_properties(ios_http_transport.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()
add_library(llamalib_common OBJECT LLM.cpp LLM_client.cpp LLM_agent.cpp completion_processor.cpp error_handling.cpp ${IOS_SSL_FILES})
LLAMALIB_COMPILE_DEFS(llamalib_common)

# build architecture library
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(ARCHITECTURE_LIBRARY_TYPE STATIC)
else()
    set(ARCHITECTURE_LIBRARY_TYPE SHARED)
endif()

add_library(
    ${LLAMALIB_ARCHITECTURE_LIBRARY}
    ${ARCHITECTURE_LIBRARY_TYPE}
    LLM_service.cpp $<TARGET_OBJECTS:llamalib_common>
     ${LLAMA_CPP_ROOT}/tools/server/server-queue.cpp
     ${LLAMA_CPP_ROOT}/tools/server/server-common.cpp
     ${LLAMA_CPP_ROOT}/tools/server/server-task.cpp
     ${LLAMA_CPP_ROOT}/tools/server/server-http.cpp
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE -Wl,--no-undefined)
endif()
LLAMALIB_COMPILE_DEFS(${LLAMALIB_ARCHITECTURE_LIBRARY})
target_include_directories(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE "${LLAMA_CPP_ROOT}/src" "${LLAMA_CPP_ROOT}/ggml/include" "${LLAMA_CPP_ROOT}/tools/server" "${LLAMA_CPP_ROOT}/tools/mtmd" "${LLAMA_CPP_ROOT}/common")
target_link_libraries(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE common mtmd)
install(TARGETS ${LLAMALIB_ARCHITECTURE_LIBRARY} LIBRARY)

# build runtime library
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND ARCHITECTURE MATCHES "arm64")
        # ARM CPU detection via getauxval(AT_HWCAP) — no x86 CPUID available
        set(ARCHCHECKER_FILES archchecker_arm.cpp)
    elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # x86 CPU detection via CPUID
        set(ARCHCHECKER_FILES archchecker.cpp $<TARGET_OBJECTS:feature_detector>)
        set(ARCHCHECKER_INCLUDES "${CMAKE_SOURCE_DIR}/third_party/FeatureDetector/src/x86")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        # Android only needs the shared runtime dispatcher; no static library
        set(RUNTIME_LIBRARY_PAIRS "SHARED:${LLAMALIB_RUNTIME_LIBRARY}")
    else()
        set(RUNTIME_LIBRARY_PAIRS "SHARED:${LLAMALIB_RUNTIME_LIBRARY}" "STATIC:${LLAMALIB_STATIC_RUNTIME_LIBRARY}")
    endif()

    foreach(pair IN ITEMS ${RUNTIME_LIBRARY_PAIRS})
        string(REPLACE ":" ";" split_pair ${pair})
        list(GET split_pair 0 lib_type)
        list(GET split_pair 1 lib_name)

        add_library(${lib_name} ${lib_type} LLM_runtime.cpp ${ARCHCHECKER_FILES} $<TARGET_OBJECTS:llamalib_common>)
        LLAMALIB_COMPILE_DEFS(${lib_name})
        target_include_directories(${lib_name} PRIVATE ${ARCHCHECKER_INCLUDES})
        install(TARGETS ${lib_name} LIBRARY)
    endforeach()

    # merge OpenSSL into the static library (desktop only)
    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
        add_custom_command(TARGET ${LLAMALIB_STATIC_RUNTIME_LIBRARY} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/temp_extract
            COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/temp_extract && ${CMAKE_AR} x $<TARGET_FILE:cpp-httplib>
            COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/temp_extract && ${CMAKE_AR} x $<TARGET_FILE:OpenSSL::SSL>
            COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/temp_extract && ${CMAKE_AR} x $<TARGET_FILE:OpenSSL::Crypto>
            COMMAND ${CMAKE_AR} rcs $<TARGET_FILE:${LLAMALIB_STATIC_RUNTIME_LIBRARY}> ${CMAKE_CURRENT_BINARY_DIR}/temp_extract/*.o
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/temp_extract
        )
    endif()
endif()
