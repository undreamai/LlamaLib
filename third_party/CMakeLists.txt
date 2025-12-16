################################################ FeatureDetector ################################################

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(feature_detector OBJECT FeatureDetector/src/x86/cpu_x86.cpp FeatureDetector/src/x86/cpu_x86.h FeatureDetector/src/x86/cpu_x86_Linux.ipp FeatureDetector/src/x86/cpu_x86_Windows.ipp)
endif()

################################################ LLAMA.CPP FLAGS ################################################

SET(LLAMA_BUILD_COMMON ON)
SET(GGML_STATIC ON)
SET(LLAMA_BUILD_TESTS OFF)
SET(LLAMA_BUILD_EXAMPLES OFF)
SET(LLAMA_CURL OFF)
SET(GGML_NATIVE OFF)
SET(LLAMA_SERVER_SSL ON)
SET(LLAMA_OPENSSL ON)
SET(LLAMA_HTTPLIB ON)

if(ARCHITECTURE MATCHES "noavx")
    SET(GGML_AVX OFF)
    SET(GGML_AVX2 OFF)
    SET(GGML_FMA OFF)
elseif(ARCHITECTURE MATCHES "avx512")
    SET(GGML_AVX512 ON)
elseif(ARCHITECTURE MATCHES "avx2")
    # Nothing to define
elseif(ARCHITECTURE MATCHES "avx")
    SET(GGML_AVX2 OFF)
elseif(ARCHITECTURE MATCHES "vulkan")
    SET(GGML_VULKAN ON)
elseif(ARCHITECTURE MATCHES "tinyblas")
    SET(GGML_CUDA ON)
    set(GGML_CUDA_FA OFF)
    set(GGML_CUDA_NO_IQUANTS ON)
    set(GGML_USE_TINYBLAS ON)
elseif(ARCHITECTURE MATCHES "cublas")
    SET(GGML_CUDA ON)
elseif(ARCHITECTURE MATCHES "hip")
    SET(GGML_HIP ON)
    SET(GGML_HIP_ROCWMMA_FATTN ON)
    SET(AMDGPU_TARGETS "gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102")
elseif(ARCHITECTURE MATCHES "acc")
    if(ARCHITECTURE MATCHES "no-acc")
        SET(GGML_ACCELERATE OFF)
        SET(GGML_BLAS OFF)
    endif()
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3")
    if(ARCHITECTURE MATCHES "arm64")
        add_compile_options("-march=armv8.2-a")
        SET(GGML_METAL_EMBED_LIBRARY ON)
    else()
        add_compile_options("-march=x86-64-v3")
        SET(GGML_METAL OFF)
    endif()
elseif(ARCHITECTURE MATCHES "android")
    SET(GGML_OPENMP OFF)
    if(ARCHITECTURE MATCHES "arm64")
        add_compile_options("-march=armv8-a")
    else()
        add_compile_options("-march=x86-64")
    endif()
elseif(ARCHITECTURE MATCHES "ios" OR ARCHITECTURE MATCHES "visionos")
    SET(GGML_METAL_EMBED_LIBRARY ON)
elseif(ARCHITECTURE STREQUAL "")
    message(FATAL_ERROR "Need to define architecture (ARCHITECTURE)")
else()
    message(FATAL_ERROR "Unknown architecture: ${ARCHITECTURE}")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    SET(LLAMA_FATAL_WARNINGS ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT DEFINED GGML_OPENMP OR GGML_OPENMP)
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -print-file-name=libgomp.so
            OUTPUT_VARIABLE GOMP_LIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(OpenMP_gomp_LIBRARY ${GOMP_LIB})
    endif()
endif()

################################################ SETUP LLAMA.CPP ################################################

# apply patches
function(APPLY_PATCH PATCH ROOT)
    get_filename_component(PATCHNAME ${PATCH} NAME)
    set(PATCH_MARKER "${ROOT}/${PATCHNAME}")
    if(NOT EXISTS ${PATCH_MARKER})
        execute_process(
            COMMAND git apply ${PATCH}
            WORKING_DIRECTORY ${ROOT}
            RESULT_VARIABLE result
        )

        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${PATCH}")
        else()
            file(WRITE ${PATCH_MARKER} "${PATCH} patch applied")
            message(STATUS "${PATCH} patch applied")
        endif()
    endif()
endfunction()

APPLY_PATCH(${CMAKE_SOURCE_DIR}/patches/llama.cpp.patch ${LLAMA_CPP_ROOT})
if(ARCHITECTURE MATCHES "tinyblas")
    APPLY_PATCH(${CMAKE_SOURCE_DIR}/patches/tinyBLAS.patch ${LLAMA_CPP_ROOT})
endif()

# copy include files needed into LlamaLib
file(COPY ${LLAMA_CPP_ROOT}/vendor/nlohmann/json.hpp DESTINATION ${CMAKE_SOURCE_DIR}/include)
file(COPY ${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.h DESTINATION ${CMAKE_SOURCE_DIR}/include)

set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

# generate files needed by llama.cpp
foreach(asset ${PUBLIC_ASSETS})
    set(input "${LLAMA_CPP_ROOT}/tools/server/public/${asset}")
    set(output "${LLAMA_CPP_ROOT}/tools/server/${asset}.hpp")
    list(APPEND GENERATED_HEADERS ${output})

    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${LLAMA_CPP_ROOT}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_custom_target(generate_assets ALL DEPENDS ${GENERATED_HEADERS})

################################################ BUILD ################################################

execute_process(
    COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY ${LLAMA_CPP_ROOT}
    OUTPUT_VARIABLE LLAMA_BUILD_NUMBER
)
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

add_subdirectory("llama.cpp")
add_subdirectory("llama.cpp/tools/mtmd")

# Output override
set_property(DIRECTORY llama.cpp PROPERTY CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if (NOT WIN32)
    foreach (target common ggml-base ggml-cpu ggml-amx ggml-blas ggml-cann ggml-cuda ggml-hip ggml-kompute ggml-metal ggml-musa ggml-rpc ggml-sycl ggml-vulkan ggml llama)
        if (TARGET ${target})
            set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${STATIC_FLAGS})
            target_compile_options(${target} PRIVATE -Wno-return-type -Wno-sometimes-uninitialized)
            target_compile_definitions(${target} PUBLIC NDEBUG)
        endif()
    endforeach()
endif()

if(ARCHITECTURE MATCHES "tinyblas")
    foreach (target common ggml-cuda ggml-hip ggml-musa)
        if (TARGET ${target})
            target_compile_definitions(${target} PUBLIC GGML_USE_TINYBLAS GGML_CUDA_NO_FA GGML_CUDA_NO_IQUANTS)
        endif()
    endforeach()
endif()