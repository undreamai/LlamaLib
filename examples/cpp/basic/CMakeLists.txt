cmake_minimum_required(VERSION 3.22)
project(LlamaLibExample)

# Find LlamaLib package
find_package(LLAMALIB REQUIRED)

# Create your executable
add_executable(main main.cpp)

# Link against LlamaLib
target_link_libraries(main PRIVATE ${LLAMALIB_LIBRARIES})
