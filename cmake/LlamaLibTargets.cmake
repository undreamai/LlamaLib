# LlamaLib Targets Configuration
# This file creates the imported targets for LlamaLib

message(STATUS "===== LlamaLib =====")
message(STATUS "LlamaLib version: ${LLAMALIB_VERSION}")

# Determine platform and architecture
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LLAMALIB_PLATFORM "win")
    set(LLAMALIB_ARCH "x64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LLAMALIB_PLATFORM "linux")
    set(LLAMALIB_ARCH "x64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LLAMALIB_PLATFORM "osx")
    # Determine macOS architecture
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(LLAMALIB_ARCH "arm64")
    else()
        set(LLAMALIB_ARCH "x64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(LLAMALIB_PLATFORM "android")
    # Determine Android architecture
    if(LLAMALIB_ANDROID_X64)
        set(LLAMALIB_ARCH "x64")
    else()
        set(LLAMALIB_ARCH "arm64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMALIB_PLATFORM "ios")
    set(LLAMALIB_ARCH "arm64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(LLAMALIB_PLATFORM "visionos")
    set(LLAMALIB_ARCH "arm64")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Set runtime identifier and library directory
set(LLAMALIB_RID "${LLAMALIB_PLATFORM}-${LLAMALIB_ARCH}")
set(LLAMALIB_LIB_REL_DIR "runtimes/${LLAMALIB_RID}/native")
set(LLAMALIB_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/${LLAMALIB_LIB_REL_DIR}")

message(STATUS "Runtime Identifier: ${LLAMALIB_RID}")
message(STATUS "LlamaLib include dir: ${LLAMALIB_INCLUDE_DIRS}")
message(STATUS "LlamaLib library dir: ${LLAMALIB_LIB_DIR}")

set(LlamaLib_LIBRARIES)

# Function to find and create imported target
function(create_llamalib_target TARGET_NAME LIB_VARIANT)
    # Prevent multiple definitions
    if(TARGET ${TARGET_NAME})
        return()
    endif()
    
    # Construct library name based on platform and variant
    set(VARIANT_NAME "llamalib_${LLAMALIB_RID}")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(VARIANT_NAME "lib${VARIANT_NAME}")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(VARIANT_NAME "${VARIANT_NAME}_${LIB_VARIANT}")
        if(LIB_VARIANT STREQUAL "runtime")
            set(VARIANT_NAME "${VARIANT_NAME}_static")
        endif()
    endif()

    if(LIB_VARIANT STREQUAL "runtime" OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
        set(LIB_TYPE "STATIC")
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(LIB_SUFFIX "lib")
        else()
            set(LIB_SUFFIX "a")
        endif()
    else()
        set(LIB_TYPE "SHARED")
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(LIB_SUFFIX "dll")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(LIB_SUFFIX "dylib")
        else()
            set(LIB_SUFFIX "so")
        endif()
    endif()

    set(FULL_LIB_NAME "${VARIANT_NAME}.${LIB_SUFFIX}")    
    set(LIB_PATH "${LLAMALIB_LIB_DIR}/${FULL_LIB_NAME}")
    
    if(EXISTS "${LIB_PATH}")
        if(LIB_TYPE STREQUAL "SHARED" AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
            # For Windows DLLs, check for import library
            string(REPLACE ".dll" ".lib" IMPORT_LIB_PATH "${LIB_PATH}")
            if(EXISTS "${IMPORT_LIB_PATH}")
                add_library(${TARGET_NAME} SHARED IMPORTED)
                set_target_properties(${TARGET_NAME} PROPERTIES
                    IMPORTED_LOCATION "${LIB_PATH}"
                    IMPORTED_IMPLIB "${IMPORT_LIB_PATH}"
                )
            else()
                message(WARNING "Import library not found for ${LIB_VARIANT}: ${IMPORT_LIB_PATH}")
                return()
            endif()
        else()
            # Static or shared library (non-Windows)
            add_library(${TARGET_NAME} ${LIB_TYPE} IMPORTED)
            set_target_properties(${TARGET_NAME} PROPERTIES
                IMPORTED_LOCATION "${LIB_PATH}"
            )
        endif()
        
        # Set include directories
        target_include_directories(${TARGET_NAME} INTERFACE "${LLAMALIB_INCLUDE_DIRS}" )
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            target_compile_definitions(${TARGET_NAME} INTERFACE WIN32_LEAN_AND_MEAN)
        endif()

        if(LLAMALIB_COPY_DEPS AND LIB_TYPE STREQUAL "SHARED")
            # automatically copy shared libraries on build
            add_custom_target("COPY_EXTERNAL_${LIB_VARIANT}")
            add_custom_command(TARGET "COPY_EXTERNAL_${LIB_VARIANT}" POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${LLAMALIB_LIB_REL_DIR}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIB_PATH} "${CMAKE_CURRENT_BINARY_DIR}/${LLAMALIB_LIB_REL_DIR}"
                COMMENT "Copying LlamaLib library: ${TARGET_NAME}"
            )
            add_dependencies(${TARGET_NAME} "COPY_EXTERNAL_${LIB_VARIANT}")

            # automatically copy dependencies of shared libraries on build
            set(LIB_DEPENDENCIES)
            if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND TARGET LlamaLib::CUBLAS)
                list(APPEND LIB_DEPENDENCIES "cublas")
                list(APPEND LIB_DEPENDENCIES "cudart")
            endif()
            if(TARGET LlamaLib::VULKAN)
                list(APPEND LIB_DEPENDENCIES "vulkan")
            endif()

            foreach(DEPENDENCY ${LIB_DEPENDENCIES})
                file(GLOB DEP_FILES "${LLAMALIB_LIB_DIR}/*${DEPENDENCY}*")
                list(FILTER DEP_FILES EXCLUDE REGEX ".*llamalib.*")
                foreach(DEP_FILE ${DEP_FILES})
                    get_filename_component(DEP_NAME ${DEP_FILE} NAME)
                    add_custom_command(TARGET "COPY_EXTERNAL_${LIB_VARIANT}" POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DEP_FILE} "${CMAKE_CURRENT_BINARY_DIR}/${LLAMALIB_LIB_REL_DIR}"
                        COMMENT "Copying ${DEPENDENCY} dependency: ${DEP_NAME}"
                    )
                endforeach()
            endforeach()
        endif()

        message(STATUS "Found ${TARGET_NAME}: ${LIB_PATH}")
        set(${TARGET_NAME}_FOUND TRUE PARENT_SCOPE)
        set(LlamaLib_${LIB_VARIANT}_FOUND TRUE PARENT_SCOPE)
        set(LlamaLib_LIBRARIES ${LlamaLib_LIBRARIES} ${TARGET_NAME} PARENT_SCOPE)
    else()
        message(STATUS "Library not found: ${LIB_PATH}")
        set(${TARGET_NAME}_FOUND FALSE PARENT_SCOPE)
        set(LlamaLib_${LIB_VARIANT}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# Create targets based on enabled options
set(LlamaLib_LIBRARIES)

# Create runtime target (always available on all platforms)
if(LLAMALIB_ENABLE_RUNTIME)
    create_llamalib_target(LlamaLib::Runtime "runtime")
endif()

# Create platform-specific targets
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(LLAMALIB_ENABLE_CUBLAS)
        create_llamalib_target(LlamaLib::CUBLAS "cublas")
    endif()
    if(LLAMALIB_ENABLE_TINYBLAS)
        create_llamalib_target(LlamaLib::TINYBLAS "tinyblas")
    endif()
    if(LLAMALIB_ENABLE_HIP)
        create_llamalib_target(LlamaLib::TINYBLAS "tinyblas")
    endif()
    if(LLAMALIB_ENABLE_VULKAN)
        create_llamalib_target(LlamaLib::VULKAN "vulkan")
    endif()
    if(LLAMALIB_ENABLE_AVX512)
        create_llamalib_target(LlamaLib::AVX512 "avx512")
    endif()
    if(LLAMALIB_ENABLE_AVX2)
        create_llamalib_target(LlamaLib::AVX2 "avx2")
    endif()
    if(LLAMALIB_ENABLE_AVX)
        create_llamalib_target(LlamaLib::AVX "avx")
    endif()
    if(LLAMALIB_ENABLE_NOAVX)
        create_llamalib_target(LlamaLib::NOAVX "noavx")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(LLAMALIB_ENABLE_ACCELERATE)
        create_llamalib_target(LlamaLib::ACCELERATE "acc")
    endif()
    if(LLAMALIB_ENABLE_NO_ACCELERATE)
        create_llamalib_target(LlamaLib::NOACCELERATE "no-acc")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    create_llamalib_target(LlamaLib::Android "android")
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    create_llamalib_target(LlamaLib::iOS "ios")
elseif(CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    create_llamalib_target(LlamaLib::VisionOS "visionos")
endif()

message(STATUS "LlamaLib libraries: ${LlamaLib_LIBRARIES}")

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