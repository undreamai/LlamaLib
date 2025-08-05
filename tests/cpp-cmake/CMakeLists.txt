cmake_minimum_required(VERSION 3.22)
project(TestCMake)

set(RUNTIME_TESTS "" CACHE STRING "Use runtime")

find_package(LlamaLib REQUIRED)

set(NAME "test")
if (RUNTIME_TESTS)
    set(NAME "test_runtime")
endif()

add_executable(${NAME} test.cpp)

if (RUNTIME_TESTS)
    target_compile_definitions(${NAME} PRIVATE RUNTIME_TESTS)
    target_link_libraries(${NAME} PRIVATE ${LlamaLib_LIBRARIES})
else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_libraries(${NAME} PRIVATE LlamaLib::NOACCELERATE)
    elseif(LlamaLib::AVX2_FOUND)
        target_link_libraries(${NAME} PRIVATE LlamaLib::AVX2)
    else()
        target_link_libraries(${NAME} PRIVATE LlamaLib::NOAVX)
    endif()
endif()
