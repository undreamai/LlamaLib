cmake_minimum_required(VERSION 3.4)
project(ArchChecker CXX)

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_subdirectory(FeatureDetector/src/x86)

set(TARGET archchecker)
add_library(${TARGET} SHARED archchecker.cpp)
target_link_libraries(${TARGET} PUBLIC FeatureDetector)
add_compile_definitions(${TARGET} PRIVATE UNDREAMAI_EXPORTS)
