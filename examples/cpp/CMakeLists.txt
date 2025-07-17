cmake_minimum_required(VERSION 3.22)
project(MyLlamaProject)

# Find LlamaLib package
find_package(LlamaLib REQUIRED)

# Create your executable
add_executable(main main.cpp)

# Link against LlamaLib
target_link_libraries(main PRIVATE ${LlamaLib_LIBRARIES})
