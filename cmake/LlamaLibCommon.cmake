function(find_path_in_current_or_parent OUT_VAR REL_PATH)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${REL_PATH}")
        set(${OUT_VAR} "${CMAKE_CURRENT_LIST_DIR}/${REL_PATH}" PARENT_SCOPE)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../${REL_PATH}")
        set(${OUT_VAR} "${CMAKE_CURRENT_LIST_DIR}/../${REL_PATH}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Required path '${REL_PATH}' not found in current or parent directory.")
    endif()
endfunction()

# Example for VERSION file
find_path_in_current_or_parent(VERSION_PATH "VERSION")
file(READ "${VERSION_PATH}" LLAMALIB_VERSION)
string(STRIP "${LLAMALIB_VERSION}" LLAMALIB_VERSION)

# Example for include dirs
set(include_paths
    "include"
    "third_party/llama.cpp/include"
    "third_party/llama.cpp/common"
    "third_party/llama.cpp/ggml/include"
    "third_party/llama.cpp/examples/server"
    "third_party/FeatureDetector/src/x86"
)

set(LLAMALIB_INCLUDE_DIRS "")
foreach(rel_path IN LISTS include_paths)
    find_path_in_current_or_parent(abs_path "${rel_path}")
    list(APPEND LLAMALIB_INCLUDE_DIRS "${abs_path}")
endforeach()
