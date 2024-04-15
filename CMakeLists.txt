cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("undreamai" C CXX)
add_subdirectory("llama.cpp")

set(TARGET undreamai)
add_library(${TARGET} undreamai.cpp)
install(TARGETS ${TARGET} LIBRARY)
target_include_directories(${TARGET} PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR} llama.cpp/examples/server)
target_link_libraries(${TARGET} PRIVATE common llama json-schema-to-grammar  ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
