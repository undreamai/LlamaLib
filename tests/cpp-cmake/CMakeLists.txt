cmake_minimum_required(VERSION 3.22)
project(TestCMake)

find_package(LLAMALIB REQUIRED)

set(NAME "test")
add_executable(${NAME} test.cpp)
target_link_libraries(${NAME} PRIVATE ${LLAMALIB_LIBRARIES})