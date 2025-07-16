
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

# Set default options
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(NOT DEFINED LLAMALIB_ENABLE_RUNTIME)
      OPTION(LLAMALIB_ENABLE_RUNTIME "Enable runtime detection" ON)
  endif()
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  option(LLAMALIB_ALLOW_GPU "Enable GPU support" ON)
  # GPU-specific options (only if GPU is enabled)
  if(LLAMALIB_ALLOW_GPU)
    if(NOT DEFINED LLAMALIB_ENABLE_CUBLAS)
        OPTION(LLAMALIB_ENABLE_CUBLAS "Enable CUBLAS architecture (GPU)" ON)
    endif()
    if(NOT DEFINED LLAMALIB_ENABLE_TINYBLAS)
        OPTION(LLAMALIB_ENABLE_TINYBLAS "Enable tinyBLAS architecture (GPU)" ON)
    endif()
    if(NOT DEFINED LLAMALIB_ENABLE_VULKAN)
        OPTION(LLAMALIB_ENABLE_VULKAN "Enable Vulkan architecture (GPU)" ON)
    endif()
    if(NOT DEFINED LLAMALIB_ENABLE_HIP)
        OPTION(LLAMALIB_ENABLE_HIP "Enable HIP architecture (GPU)" ON)
    endif()
  else()
      # Force GPU options to OFF when GPU is disabled
      set(LLAMALIB_ENABLE_CUBLAS OFF CACHE BOOL "Enable CUBLAS architecture (GPU)" FORCE)
      set(LLAMALIB_ENABLE_TINYBLAS OFF CACHE BOOL "Enable tinyBLAS architecture (GPU)" FORCE)
      set(LLAMALIB_ENABLE_VULKAN OFF CACHE BOOL "Enable Vulkan architecture (GPU)" FORCE)
      set(LLAMALIB_ENABLE_HIP OFF CACHE BOOL "Enable HIP architecture (GPU)" FORCE)
  endif()

  if(NOT DEFINED LLAMALIB_ENABLE_AVX512)
      OPTION(LLAMALIB_ENABLE_AVX512 "Enable AVX512 architecture" ON)
  endif()
  if(NOT DEFINED LLAMALIB_ENABLE_AVX2)
      OPTION(LLAMALIB_ENABLE_AVX2 "Enable AVX2 architecture" ON)
  endif()
  if(NOT DEFINED LLAMALIB_ENABLE_AVX)
      OPTION(LLAMALIB_ENABLE_AVX "Enable AVX architecture" ON)
  endif()
  if(NOT DEFINED LLAMALIB_ENABLE_NOAVX)
      OPTION(LLAMALIB_ENABLE_NOAVX "Enable no-AVX architecture" ON)
  endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(NOT DEFINED LLAMALIB_ENABLE_ACCELERATE)
      OPTION(LLAMALIB_ENABLE_ACCELERATE "Enable architecture with Accelerate framework" ON)
  endif()
  if(NOT DEFINED LLAMALIB_ENABLE_NO_ACCELERATE)
      OPTION(LLAMALIB_ENABLE_NO_ACCELERATE "Enable architecture without Accelerate framework" ON)
  endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  if(NOT DEFINED LLAMALIB_ANDROID_X64)
      OPTION(LLAMALIB_ANDROID_X64 "Use Android X64 instead of ARM64 architecture" ON)
  endif()
endif()

if(NOT DEFINED LLAMALIB_COPY_DEPS)
    OPTION(LLAMALIB_COPY_DEPS "Automatically copy LlamaLib libraries" ON)
endif()

# Include the main logic from a separate module
include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibTargets.cmake")
