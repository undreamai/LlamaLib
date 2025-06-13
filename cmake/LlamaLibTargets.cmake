# LlamaLib Targets Configuration
# This file creates the imported targets for LlamaLib

message(STATUS "=== LlamaLib ===")
message(STATUS "LlamaLib version: ${LLAMALIB_VERSION}")
message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# Determine platform and architecture
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LLAMALIB_PLATFORM "windows")
    set(LLAMALIB_LIB_PREFIX "llamalib_windows_")
    set(LLAMALIB_LIB_SUFFIX ".dll")
    set(LLAMALIB_IMPORT_SUFFIX ".lib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LLAMALIB_PLATFORM "linux")
    set(LLAMALIB_LIB_PREFIX "libllamalib_linux_")
    set(LLAMALIB_LIB_SUFFIX ".so")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LLAMALIB_PLATFORM "macos")
    set(LLAMALIB_LIB_PREFIX "libllamalib_macos_")
    set(LLAMALIB_LIB_SUFFIX ".dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(LLAMALIB_PLATFORM "android") 
    set(LLAMALIB_LIB_PREFIX "libllamalib_android_")
    set(LLAMALIB_LIB_SUFFIX ".so")
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMALIB_PLATFORM "ios")
    set(LLAMALIB_LIB_PREFIX "libllamalib_")
    set(LLAMALIB_LIB_SUFFIX ".a")
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(LLAMALIB_PLATFORM "visionos")
    set(LLAMALIB_LIB_PREFIX "libllamalib_")
    set(LLAMALIB_LIB_SUFFIX ".a")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Set library directory
set(LLAMALIB_LIB_DIR "${LLAMALIB_LIB_DIR}/${LLAMALIB_PLATFORM}")
message(STATUS "Platform: ${LLAMALIB_PLATFORM}")
message(STATUS "LlamaLib include dir: ${LLAMALIB_INCLUDE_DIRS}")
message(STATUS "LlamaLib library dir: ${LLAMALIB_LIB_DIR}")

# Function to find and create imported target
function(create_llamalib_target TARGET_NAME LIB_NAME)
    # Prevent multiple definitions
    if(TARGET ${TARGET_NAME})
        return()
    endif()
    
    # Construct library name based on platform
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        if(LLAMALIB_ENABLE_ANDROID_X64)
            set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}x64${LLAMALIB_LIB_SUFFIX}")
        else()
            set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}arm64${LLAMALIB_LIB_SUFFIX}")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}ios${LLAMALIB_LIB_SUFFIX}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
        set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}visionos${LLAMALIB_LIB_SUFFIX}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT LIB_NAME STREQUAL "runtime")
        # macOS acceleration libraries
        if(LLAMALIB_ENABLE_ACCELERATION)
            set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}${LLAMALIB_ARCH}_acc${LLAMALIB_LIB_SUFFIX}")
        else()
            set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}${LLAMALIB_ARCH}_no_acc${LLAMALIB_LIB_SUFFIX}")
        endif()
    else()
        set(FULL_LIB_NAME "${LLAMALIB_LIB_PREFIX}${LIB_NAME}${LLAMALIB_LIB_SUFFIX}")
    endif()
    
    set(LIB_PATH "${LLAMALIB_LIB_DIR}/${FULL_LIB_NAME}")
    
    if(EXISTS "${LIB_PATH}")
        if(LLAMALIB_LIB_SUFFIX STREQUAL ".dll")
            # For Windows, we need both DLL and import library
            string(REPLACE ".dll" ".lib" IMPORT_LIB_PATH "${LIB_PATH}")
            if(EXISTS "${IMPORT_LIB_PATH}")
                add_library(${TARGET_NAME} SHARED IMPORTED)
                set_target_properties(${TARGET_NAME} PROPERTIES
                    IMPORTED_LOCATION "${LIB_PATH}"
                    IMPORTED_IMPLIB "${IMPORT_LIB_PATH}"
                )
            else()
                message(WARNING "Import library not found for ${LIB_NAME}: ${IMPORT_LIB_PATH}")
                return()
            endif()
        elseif(LLAMALIB_LIB_SUFFIX STREQUAL ".a")
            # Static library
            add_library(${TARGET_NAME} STATIC IMPORTED)
            set_target_properties(${TARGET_NAME} PROPERTIES
                IMPORTED_LOCATION "${LIB_PATH}"
            )
        else()
            # Shared library
            add_library(${TARGET_NAME} SHARED IMPORTED)
            set_target_properties(${TARGET_NAME} PROPERTIES
                IMPORTED_LOCATION "${LIB_PATH}"
            )
        endif()
        
        # Set include directories
        target_include_directories(${TARGET_NAME} INTERFACE
            "${LLAMALIB_INCLUDE_DIRS}"
        )
        
        message(STATUS "Found ${TARGET_NAME}: ${LIB_PATH}")
        set(${TARGET_NAME}_FOUND TRUE PARENT_SCOPE)
        set(LlamaLib_${LIB_NAME}_FOUND TRUE PARENT_SCOPE)
    else()
        message(STATUS "Library not found: ${LIB_PATH}")
        set(${TARGET_NAME}_FOUND FALSE PARENT_SCOPE)
        set(LlamaLib_${LIB_NAME}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# Create targets based on enabled options
set(LlamaLib_LIBRARIES)

# Create runtime target (always available on all platforms)
if(LLAMALIB_ENABLE_RUNTIME)
    create_llamalib_target(LlamaLib::Runtime "runtime")
    if(LlamaLib_runtime_FOUND)
        list(APPEND LlamaLib_LIBRARIES LlamaLib::Runtime)
    endif()
endif()

# Create platform-specific targets
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(LLAMALIB_ENABLE_AVX)
        create_llamalib_target(LlamaLib::AVX "avx")
        if(LlamaLib_avx_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX)
        endif()
    endif()
    if(LLAMALIB_ENABLE_AVX2)
        create_llamalib_target(LlamaLib::AVX2 "avx2")
        if(LlamaLib_avx2_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX2)
        endif()
    endif()
    if(LLAMALIB_ENABLE_AVX512)
        create_llamalib_target(LlamaLib::AVX512 "avx512")
        if(LlamaLib_avx512_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX512)
        endif()
    endif()
    if(LLAMALIB_ENABLE_NOAVX)
        create_llamalib_target(LlamaLib::NoAVX "noavx")
        if(LlamaLib_noavx_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::NoAVX)
        endif()
    endif()
    if(LLAMALIB_ENABLE_CUBLAS)
        create_llamalib_target(LlamaLib::CUBLAS "cublas")
        if(LlamaLib_cublas_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::CUBLAS)
        endif()
    endif()
    if(LLAMALIB_ENABLE_TINYBLAS)
        create_llamalib_target(LlamaLib::TinyBLAS "tinyblas")
        if(LlamaLib_tinyblas_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::TinyBLAS)
        endif()
    endif()
    if(LLAMALIB_ENABLE_VULKAN)
        create_llamalib_target(LlamaLib::Vulkan "vulkan")
        if(LlamaLib_vulkan_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::Vulkan)
        endif()
    endif()
    if(LLAMALIB_ENABLE_HIP)
        create_llamalib_target(LlamaLib::HIP "hip")
        if(LlamaLib_hip_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::HIP)
        endif()
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(LLAMALIB_ENABLE_AVX)
        create_llamalib_target(LlamaLib::AVX "avx")
        if(LlamaLib_avx_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX)
        endif()
    endif()
    if(LLAMALIB_ENABLE_AVX2)
        create_llamalib_target(LlamaLib::AVX2 "avx2")
        if(LlamaLib_avx2_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX2)
        endif()
    endif()
    if(LLAMALIB_ENABLE_AVX512)
        create_llamalib_target(LlamaLib::AVX512 "avx512")
        if(LlamaLib_avx512_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::AVX512)
        endif()
    endif()
    if(LLAMALIB_ENABLE_NOAVX)
        create_llamalib_target(LlamaLib::NoAVX "noavx")
        if(LlamaLib_noavx_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::NoAVX)
        endif()
    endif()
    if(LLAMALIB_ENABLE_CUBLAS)
        create_llamalib_target(LlamaLib::CUBLAS "cublas")
        if(LlamaLib_cublas_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::CUBLAS)
        endif()
    endif()
    if(LLAMALIB_ENABLE_TINYBLAS)
        create_llamalib_target(LlamaLib::TinyBLAS "tinyblas")
        if(LlamaLib_tinyblas_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::TinyBLAS)
        endif()
    endif()
    if(LLAMALIB_ENABLE_VULKAN)
        create_llamalib_target(LlamaLib::Vulkan "vulkan")
        if(LlamaLib_vulkan_FOUND)
            list(APPEND LlamaLib_LIBRARIES LlamaLib::Vulkan)
        endif()
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS has acceleration variants
    create_llamalib_target(LlamaLib::Acceleration "acceleration")
    if(LlamaLib_acceleration_FOUND)
        list(APPEND LlamaLib_LIBRARIES LlamaLib::Acceleration)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android libraries are arch-specific
    create_llamalib_target(LlamaLib::Android "android")
    if(LlamaLib_android_FOUND)
        list(APPEND LlamaLib_LIBRARIES LlamaLib::Android)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # iOS has universal library
    create_llamalib_target(LlamaLib::iOS "ios")
    if(LlamaLib_ios_FOUND)
        list(APPEND LlamaLib_LIBRARIES LlamaLib::iOS)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    # visionOS has universal library
    create_llamalib_target(LlamaLib::VisionOS "visionos")
    if(LlamaLib_visionos_FOUND)
        list(APPEND LlamaLib_LIBRARIES LlamaLib::VisionOS)
    endif()
endif()

message(STATUS "LlamaLib libraries: ${LlamaLib_LIBRARIES}")

# Function for dependency copying
function(add_dependency DEPENDENCY)
    file(GLOB DEP_DLLS "${LLAMALIB_LIB_DIR}/*${DEPENDENCY}*")
    list(FILTER DEP_DLLS EXCLUDE REGEX ".*llamalib.*")
    foreach(DEP_DLL ${DEP_DLLS})
        get_filename_component(DEP_NAME ${DEP_DLL} NAME)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEP_DLL}"
            "$<TARGET_FILE_DIR:${TARGET}>/"
            COMMENT "Copying ${DEPENDENCY} dependency: ${DEP_NAME}"
        )
    endforeach()
endfunction()

# Function to copy libraries and dependencies
function(llamalib_copy_libraries TARGET)
    if(NOT LLAMALIB_COPY_DEPS)
        return()
    endif()
    
    # Helper function to copy a library if target exists
    function(copy_if_target_exists TARGET_NAME)
        if(TARGET ${TARGET_NAME})
            get_target_property(LIB_PATH ${TARGET_NAME} IMPORTED_LOCATION)
            if(LIB_PATH)
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${LIB_PATH}"
                    "$<TARGET_FILE_DIR:${TARGET}>/"
                    COMMENT "Copying ${TARGET_NAME}"
                )
            endif()
        endif()
    endfunction()
    
    # Copy all possible libraries
    copy_if_target_exists(LlamaLib::Runtime)
    copy_if_target_exists(LlamaLib::AVX)
    copy_if_target_exists(LlamaLib::AVX2)
    copy_if_target_exists(LlamaLib::AVX512)
    copy_if_target_exists(LlamaLib::NoAVX)
    copy_if_target_exists(LlamaLib::CUBLAS)
    copy_if_target_exists(LlamaLib::TinyBLAS)
    copy_if_target_exists(LlamaLib::Vulkan)
    copy_if_target_exists(LlamaLib::HIP)
    copy_if_target_exists(LlamaLib::Acceleration)
    copy_if_target_exists(LlamaLib::Android)
    copy_if_target_exists(LlamaLib::iOS)
    copy_if_target_exists(LlamaLib::VisionOS)
    
    # Copy Vulkan dependencies if Vulkan is enabled
    if(LLAMALIB_ENABLE_RUNTIME AND TARGET LlamaLib::Runtime)
        add_dependency("archchecker")
    endif()

    # Copy CUBLAS dependencies if CUBLAS is enabled
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND LLAMALIB_ENABLE_CUBLAS AND TARGET LlamaLib::CUBLAS)
        add_dependency("cublas")
        add_dependency("cudart")
    endif()
    
    # Copy Vulkan dependencies if Vulkan is enabled
    if(LLAMALIB_ENABLE_VULKAN AND TARGET LlamaLib::Vulkan)
        add_dependency("vulkan")
    endif()
endfunction()

# Set variables for find_package
set(LlamaLib_FOUND TRUE)
set(LlamaLib_LIBRARIES ${LlamaLib_LIBRARIES})

# Check required components
if(LlamaLib_FIND_COMPONENTS)
    set(LlamaLib_FOUND TRUE)
    foreach(component ${LlamaLib_FIND_COMPONENTS})
        string(TOUPPER ${component} component_upper)
        if(NOT LlamaLib_${component}_FOUND)
            set(LlamaLib_FOUND FALSE)
            if(LlamaLib_FIND_REQUIRED_${component})
                message(FATAL_ERROR "LlamaLib component ${component} not found")
            endif()
        endif()
    endforeach()
endif()

check_required_components(LlamaLib)
