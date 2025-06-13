
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was LlamaLibConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibCommon.cmake")
set(LLAMALIB_LIB_DIR "${PACKAGE_PREFIX_DIR}/libs")

# Set default options
if(NOT DEFINED LLAMALIB_ENABLE_RUNTIME)
    set(LLAMALIB_ENABLE_RUNTIME ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_AVX)
    set(LLAMALIB_ENABLE_AVX ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_AVX2)
    set(LLAMALIB_ENABLE_AVX2 ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_AVX512)
    set(LLAMALIB_ENABLE_AVX512 ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_NOAVX)
    set(LLAMALIB_ENABLE_NOAVX ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_CUBLAS)
    set(LLAMALIB_ENABLE_CUBLAS ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_TINYBLAS)
    set(LLAMALIB_ENABLE_TINYBLAS ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_VULKAN)
    set(LLAMALIB_ENABLE_VULKAN ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_HIP)
    set(LLAMALIB_ENABLE_HIP ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_ACCELERATION)
    set(LLAMALIB_ENABLE_ACCELERATION ON)
endif()
if(NOT DEFINED LLAMALIB_ENABLE_ANDROID_X64)
    set(LLAMALIB_ENABLE_ANDROID_X64 ON)
endif()
if(NOT DEFINED LLAMALIB_COPY_DEPS)
    set(LLAMALIB_COPY_DEPS ON)
endif()

# Include the main logic from a separate module
include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibTargets.cmake")
