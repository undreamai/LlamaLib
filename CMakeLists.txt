cmake_minimum_required(VERSION 3.18)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" LLAMALIB_VERSION)
string(STRIP "${LLAMALIB_VERSION}" LLAMALIB_VERSION)
project(LlamaLib VERSION ${LLAMALIB_VERSION} LANGUAGES C CXX)

################################################ OPTIONS ################################################

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    OPTION(LLAMALIB_BUILD_SERVER "Build LlamaLib server" ON)
    OPTION(LLAMALIB_BUILD_TESTS "Build LlamaLib tests" ON)
else()
    SET(LLAMALIB_BUILD_SERVER OFF)
    SET(LLAMALIB_BUILD_TESTS OFF)
endif()
set(ARCHITECTURE "" CACHE STRING "Target architecture")
SET(BUILD_SHARED_LIBS OFF)

################################################ SETUP ################################################

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} /Zm1000")

# Setup OpenSSL
if (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(OPENSSL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/include)
    set(OPENSSL_SSL_LIBRARY ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/lib/libssl.a)
    set(OPENSSL_CRYPTO_LIBRARY ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/lib/libcrypto.a)
endif()
set(OPENSSL_USE_STATIC_LIBS ON)
find_package(OpenSSL REQUIRED)

################################################ BUILD ################################################

set(LLAMA_CPP_ROOT ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)

function(LLAMALIB_COMPILE_DEFS NAME)
    add_compile_definitions(${NAME} PRIVATE UNDREAMAI_EXPORTS)
    target_include_directories(${NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_compile_definitions(${NAME} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(${NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endfunction()

if (NOT WIN32)
    set(STATIC_FLAGS "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STATIC_FLAGS}")
else()
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

string(REGEX MATCH "^[^_]+" PLATFORM "${ARCHITECTURE}")

set(LLAMALIB_ARCHITECTURE_LIBRARY "llamalib_${ARCHITECTURE}")
set(LLAMALIB_RUNTIME_LIBRARY "llamalib_${PLATFORM}_runtime")
set(LLAMALIB_STATIC_RUNTIME_LIBRARY "llamalib_${PLATFORM}_runtime_static")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    SET(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    SET(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)
    SET(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM UndreamAI)
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    SET(CMAKE_SYSTEM_NAME visionOS)
    SET(CMAKE_OSX_DEPLOYMENT_TARGET 1.0)
    SET(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM UndreamAI)
endif()

################################################ BUILD DIRS ################################################

add_subdirectory("third_party")
add_subdirectory("src")
add_subdirectory("tools")
if (LLAMALIB_BUILD_TESTS)
    add_subdirectory("tests/cpp")
endif()
