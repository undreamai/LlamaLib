cmake_minimum_required(VERSION 3.22)
project(MyLlamaProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LlamaLib package
find_package(LlamaLib REQUIRED)

# Create your executable
add_executable(main main.cpp)

add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
# Link against LlamaLib
# Option 1: Link all available targets
# target_link_libraries(main PRIVATE ${LlamaLib_LIBRARIES})

# Option 2: Link target architecture directly
# target_link_libraries(main PRIVATE LlamaLib::AVX2)

# Option 3: Link specific runtime architectures
# target_link_libraries(main PRIVATE LlamaLib::Runtime LlamaLib::CUBLAS LlamaLib::AVX2 LlamaLib::NOAVX)
target_link_libraries(main PRIVATE LlamaLib::Runtime)