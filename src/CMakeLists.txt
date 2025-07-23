add_library(llamalib_common OBJECT LLM.cpp LLM_client.cpp error_handling.cpp)
LLAMALIB_COMPILE_DEFS(llamalib_common)

# build architecture library
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(ARCHITECTURE_LIBRARY_TYPE STATIC)
else()
    set(ARCHITECTURE_LIBRARY_TYPE SHARED)
endif()

add_library(${LLAMALIB_ARCHITECTURE_LIBRARY} ${ARCHITECTURE_LIBRARY_TYPE} LLM_service.cpp $<TARGET_OBJECTS:llamalib_common>)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE -Wl,--no-undefined)
endif()
LLAMALIB_COMPILE_DEFS(${LLAMALIB_ARCHITECTURE_LIBRARY})
target_include_directories(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE "../third_party/llama.cpp/src")
target_link_libraries(${LLAMALIB_ARCHITECTURE_LIBRARY} PRIVATE common)
install(TARGETS ${LLAMALIB_ARCHITECTURE_LIBRARY} LIBRARY)

# build runtime library
if (LLAMALIB_BUILD_RUNTIME_LIB)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ARCHCHECKER_FILES archchecker.cpp $<TARGET_OBJECTS:feature_detector>)
    endif()

    foreach(pair IN ITEMS "SHARED:${LLAMALIB_RUNTIME_LIBRARY}" "STATIC:${LLAMALIB_STATIC_RUNTIME_LIBRARY}")
        string(REPLACE ":" ";" split_pair ${pair})
        list(GET split_pair 0 lib_type)
        list(GET split_pair 1 lib_name)

        add_library(${lib_name} ${lib_type} LLM_runtime.cpp ${ARCHCHECKER_FILES} $<TARGET_OBJECTS:llamalib_common>)
        LLAMALIB_COMPILE_DEFS(${lib_name})
        install(TARGETS ${lib_name} LIBRARY)
    endforeach()

    # merge OpenSSL into the static library
    if (NOT WIN32)
        add_custom_command(TARGET ${LLAMALIB_STATIC_RUNTIME_LIBRARY} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/temp_extract
            COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/temp_extract && ${CMAKE_AR} x $<TARGET_FILE:OpenSSL::SSL>
            COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/temp_extract && ${CMAKE_AR} x $<TARGET_FILE:OpenSSL::Crypto>
            COMMAND ${CMAKE_AR} rcs $<TARGET_FILE:${LLAMALIB_STATIC_RUNTIME_LIBRARY}> ${CMAKE_CURRENT_BINARY_DIR}/temp_extract/*.o
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/temp_extract
        )
    endif()
endif()
