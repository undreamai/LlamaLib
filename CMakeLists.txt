cmake_minimum_required(VERSION 3.14)
project("undreamai" C CXX)

add_subdirectory("llama.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(TARGET undreamai)
add_library(${TARGET} undreamai.cpp)
# add_executable(${TARGET} undreamai.cpp)
install(TARGETS ${TARGET} LIBRARY)
target_include_directories(${TARGET} PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR} llama.cpp/examples/server)
target_link_libraries(${TARGET} PRIVATE common json-schema-to-grammar llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
