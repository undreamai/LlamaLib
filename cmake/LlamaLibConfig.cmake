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

# List of all architecture options
set(LLAMALIB_GPU_OPTIONS
    LLAMALIB_USE_CUBLAS
    LLAMALIB_USE_TINYBLAS
    LLAMALIB_USE_HIP
    LLAMALIB_USE_VULKAN
)

set(LLAMALIB_CPU_OPTIONS
    LLAMALIB_USE_AVX512
    LLAMALIB_USE_AVX2
    LLAMALIB_USE_AVX
    LLAMALIB_USE_NOAVX
)

set(LLAMALIB_MACOS_OPTIONS
    LLAMALIB_USE_ACCELERATE
    LLAMALIB_USE_NO_ACCELERATE
)

# Combine all architecture options
set(LLAMALIB_ALL_ARCH_OPTIONS
    ${LLAMALIB_GPU_OPTIONS}
    ${LLAMALIB_CPU_OPTIONS}
    ${LLAMALIB_MACOS_OPTIONS}
)

# Check if user explicitly set any architecture options
set(LLAMALIB_USER_SPECIFIED_ARCH FALSE)
foreach(OPT ${LLAMALIB_ALL_ARCH_OPTIONS})
    if(DEFINED ${OPT})
        set(LLAMALIB_USER_SPECIFIED_ARCH TRUE)
        break()
    endif()
endforeach()

# Set default options based on platform
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Global CPU/GPU toggles
    option(LLAMALIB_ALLOW_CPU "Enable CPU support" ON)
    option(LLAMALIB_ALLOW_GPU "Enable GPU support" ON)

    # If user specified ANY architecture, use only those specified
    # Otherwise, default all to ON
    if(NOT LLAMALIB_USER_SPECIFIED_ARCH)
        # Default: all architectures ON (unless category is disabled)
        if(LLAMALIB_ALLOW_GPU)
            foreach(OPT ${LLAMALIB_GPU_OPTIONS})
                option(${OPT} "Enable ${OPT}" ON)
            endforeach()
        else()
            foreach(OPT ${LLAMALIB_GPU_OPTIONS})
                set(${OPT} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_GPU is OFF" FORCE)
            endforeach()
        endif()

        if(LLAMALIB_ALLOW_CPU)
            foreach(OPT ${LLAMALIB_CPU_OPTIONS})
                option(${OPT} "Enable ${OPT}" ON)
            endforeach()
        else()
            foreach(OPT ${LLAMALIB_CPU_OPTIONS})
                set(${OPT} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_CPU is OFF" FORCE)
            endforeach()
        endif()
    else()
        # User specified some: default unspecified to OFF
        # But still respect ALLOW_CPU/ALLOW_GPU overrides
        foreach(OPT ${LLAMALIB_GPU_OPTIONS})
            if(NOT LLAMALIB_ALLOW_GPU)
                set(${OPT} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_GPU is OFF" FORCE)
            elseif(NOT DEFINED ${OPT})
                option(${OPT} "Enable ${OPT}" OFF)
            else()
                option(${OPT} "Enable ${OPT}" ${${OPT}})
            endif()
        endforeach()

        foreach(OPT ${LLAMALIB_CPU_OPTIONS})
            if(NOT LLAMALIB_ALLOW_CPU)
                set(${OPT} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_CPU is OFF" FORCE)
            elseif(NOT DEFINED ${OPT})
                option(${OPT} "Enable ${OPT}" OFF)
            else()
                option(${OPT} "Enable ${OPT}" ${${OPT}})
            endif()
        endforeach()
    endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS: same logic for Accelerate options
    if(NOT LLAMALIB_USER_SPECIFIED_ARCH)
        # Default: all ON
        foreach(OPT ${LLAMALIB_MACOS_OPTIONS})
            option(${OPT} "Enable ${OPT}" ON)
        endforeach()
    else()
        # User specified some: default unspecified to OFF
        foreach(OPT ${LLAMALIB_MACOS_OPTIONS})
            if(NOT DEFINED ${OPT})
                option(${OPT} "Enable ${OPT}" OFF)
            else()
                option(${OPT} "Enable ${OPT}" ${${OPT}})
            endif()
        endforeach()
    endif()
endif()

# Copy dependencies option
if(NOT DEFINED LLAMALIB_COPY_DEPS)
    option(LLAMALIB_COPY_DEPS "Automatically copy LlamaLib libraries" ON)
endif()

# Report enabled architectures
set(LLAMALIB_ENABLED_ARCHS "")
foreach(OPT ${LLAMALIB_ALL_ARCH_OPTIONS})
    if(${OPT})
        list(APPEND LLAMALIB_ENABLED_ARCHS ${OPT})
    endif()
endforeach()

if(LLAMALIB_ENABLED_ARCHS)
    message(STATUS "LlamaLib enabled architectures: ${LLAMALIB_ENABLED_ARCHS}")
else()
    message(WARNING "No LlamaLib architectures enabled")
endif()

# Include the main logic from a separate module
include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibTargets.cmake")