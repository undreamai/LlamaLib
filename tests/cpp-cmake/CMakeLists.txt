cmake_minimum_required(VERSION 3.22)
project(TestCMake)
set(RUNTIME_TESTS "" CACHE STRING "Use runtime")

find_package(LlamaLib REQUIRED)

set(NAME "test")
if (RUNTIME_TESTS)
    set(NAME "test_runtime")
endif()

add_executable(${NAME} test.cpp)
target_link_libraries(${NAME} PRIVATE ${LlamaLib_LIBRARIES})

if (RUNTIME_TESTS)
    target_compile_definitions(${NAME} PRIVATE RUNTIME_TESTS)
endif()
