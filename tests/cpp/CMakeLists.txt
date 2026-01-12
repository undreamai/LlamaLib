foreach(pair IN ITEMS "llamalib_tests:${LLAMALIB_ARCHITECTURE_LIBRARY}" "llamalib_tests_runtime:${LLAMALIB_STATIC_RUNTIME_LIBRARY}")
    string(REPLACE ":" ";" split_pair ${pair})
    list(GET split_pair 0 NAME)
    list(GET split_pair 1 LLAMALIB_NAME)

    add_executable(${NAME} llamalib_tests.cpp)
    LLAMALIB_COMPILE_DEFS(${NAME})
    target_link_libraries(${NAME} PRIVATE ${LLAMALIB_NAME})

    if (NAME STREQUAL "llamalib_tests_runtime")
        target_compile_definitions(${NAME} PRIVATE USE_RUNTIME_DETECTION)
    endif()
    install(TARGETS ${NAME} RUNTIME)
endforeach()
