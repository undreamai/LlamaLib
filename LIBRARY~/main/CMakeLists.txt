cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("llmunity" C CXX)
add_subdirectory("llama.cpp")

set(TARGET llmunity)
add_library(${TARGET} llmunity.cpp)
install(TARGETS ${TARGET} LIBRARY)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET llmunity-main)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama llmunity ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
