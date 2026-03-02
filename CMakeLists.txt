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
set(CMAKE_CUDA_ARCHITECTURES "52;61;75;86;89")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} /Zm1000")

# Setup OpenSSL
set(IOS_VISIONOS_HTTP FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(IOS_VISIONOS_HTTP TRUE)
endif()

if (NOT IOS_VISIONOS_HTTP)
    if (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
        set(OPENSSL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/include)
        set(OPENSSL_SSL_LIBRARY ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/lib/libssl.a)
        set(OPENSSL_CRYPTO_LIBRARY ${CMAKE_SOURCE_DIR}/openssl/${ARCHITECTURE}/lib/libcrypto.a)
    endif()
    set(OPENSSL_USE_STATIC_LIBS ON)
    find_package(OpenSSL REQUIRED)
endif()

################################################ BUILD ################################################

set(LLAMA_CPP_ROOT ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)

if(ARCHITECTURE MATCHES "android" AND ARCHITECTURE MATCHES "vulkan")
    # Derive triple from ANDROID_ABI
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(ANDROID_ARCH_TRIPLE "aarch64-linux-android")
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(ANDROID_ARCH_TRIPLE "arm-linux-androideabi")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(ANDROID_ARCH_TRIPLE "x86_64-linux-android")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(ANDROID_ARCH_TRIPLE "i686-linux-android")
    else()
        message(FATAL_ERROR "Unknown ANDROID_ABI: ${ANDROID_ABI}")
    endif()

    message(STATUS "ANDROID_ABI=${ANDROID_ABI}")
    message(STATUS "ANDROID_ARCH_TRIPLE=${ANDROID_ARCH_TRIPLE}")
    message(STATUS "CMAKE_SYSROOT=${CMAKE_SYSROOT}")

    string(REGEX REPLACE "android-" "" ANDROID_API_LEVEL "${ANDROID_PLATFORM}")
    set(Vulkan_LIBRARY "${CMAKE_SYSROOT}/usr/lib/${ANDROID_ARCH_TRIPLE}/${ANDROID_API_LEVEL}/libvulkan.so")
    set(Vulkan_FOUND TRUE)
    set(Vulkan_VERSION "1.3.275")
    set(Vulkan_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Headers/include")

    if(NOT EXISTS "${Vulkan_LIBRARY}")
        message(FATAL_ERROR "Vulkan stub not found at: ${Vulkan_LIBRARY}")
    endif()

    find_program(Vulkan_GLSLC_EXECUTABLE
        NAMES glslc
        PATHS "/usr/bin" "/usr/local/bin"
        NO_CMAKE_SYSTEM_PATH)
    set(Vulkan_GLSLC_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}" CACHE FILEPATH "" FORCE)
endif()


function(LLAMALIB_COMPILE_DEFS NAME)
    add_compile_definitions(${NAME} PRIVATE UNDREAMAI_EXPORTS)
    target_include_directories(${NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include")
    if (IOS_VISIONOS_HTTP)
        target_link_libraries(${NAME} PRIVATE "-framework Foundation" "-framework Security""-framework CFNetwork")
        set_target_properties(${NAME} PROPERTIES XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
    else()
        target_compile_definitions(${NAME} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_link_libraries(${NAME} PRIVATE cpp-httplib OpenSSL::SSL OpenSSL::Crypto)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        target_link_options(${NAME} PRIVATE "-Wl,-z,max-page-size=16384")
    endif()
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
