if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(LIBRARY_TYPE STATIC)
else()
    set(LIBRARY_TYPE SHARED)
endif()

add_library(llamalib_common OBJECT LLM.cpp LLM_client.cpp error_handling.cpp)
LLAMALIB_COMPILE_DEFS(llamalib_common)

# build static library
add_library(${LLAMALIB_STATIC_LIBRARY} ${LIBRARY_TYPE} LLM_service.cpp $<TARGET_OBJECTS:llamalib_common>)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${LLAMALIB_STATIC_LIBRARY} PRIVATE -Wl,--no-undefined)
endif()
LLAMALIB_COMPILE_DEFS(${LLAMALIB_STATIC_LIBRARY})
target_link_libraries(${LLAMALIB_STATIC_LIBRARY} PRIVATE common)
install(TARGETS ${LLAMALIB_STATIC_LIBRARY} LIBRARY)

# build runtime library
if (LLAMALIB_BUILD_RUNTIME_LIB)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ARCHCHECKER_FILES archchecker.cpp)
    endif()

    add_library(${LLAMALIB_RUNTIME_LIBRARY} STATIC LLM_runtime.cpp ${ARCHCHECKER_FILES} $<TARGET_OBJECTS:llamalib_common>)
    LLAMALIB_COMPILE_DEFS(${LLAMALIB_RUNTIME_LIBRARY})
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(${LLAMALIB_RUNTIME_LIBRARY} PRIVATE FeatureDetector)
    endif()
    install(TARGETS ${LLAMALIB_RUNTIME_LIBRARY} LIBRARY)
endif()
