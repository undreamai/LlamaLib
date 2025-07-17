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

set(DEPENDENT_OPTIONS
    LLAMALIB_USE_CUBLAS
    LLAMALIB_USE_TINYBLAS
    LLAMALIB_USE_HIP
    LLAMALIB_USE_VULKAN
    LLAMALIB_USE_AVX512
    LLAMALIB_USE_AVX2
    LLAMALIB_USE_AVX
    LLAMALIB_USE_NOAVX
    LLAMALIB_USE_ACCELERATE
    LLAMALIB_USE_NO_ACCELERATE
)

# Enhanced llamalib_option function that creates dependent options
function(llamalib_option name description)
    if(NOT DEFINED ${name})
        set(${name} ${LLAMALIB_RUNTIME_DETECTION} CACHE BOOL "${description}")
    endif()
    
    # Create a variable to track if this option was manually overridden
    if(NOT DEFINED ${name}_MANUAL_OVERRIDE)
        set(${name}_MANUAL_OVERRIDE FALSE CACHE INTERNAL "Track if ${name} was manually set by user")
    endif()
endfunction()

# Function to handle automatic dependency management
function(handle_dependent_options)
    # Store the previous state of LLAMALIB_RUNTIME_DETECTION
    if(NOT DEFINED LLAMALIB_RUNTIME_DETECTION_PREV)
        set(LLAMALIB_RUNTIME_DETECTION_PREV ${LLAMALIB_RUNTIME_DETECTION} CACHE INTERNAL "Previous state of LLAMALIB_RUNTIME_DETECTION")
    endif()
    
    # Check if LLAMALIB_RUNTIME_DETECTION changed
    set(RUNTIME_DETECTION_CHANGED FALSE)
    if(NOT "${LLAMALIB_RUNTIME_DETECTION}" STREQUAL "${LLAMALIB_RUNTIME_DETECTION_PREV}")
        set(RUNTIME_DETECTION_CHANGED TRUE)
    endif()
    
    # List of all dependent options
    set(VALUE_AUTO_CHANGED OFF)
    foreach(OPTION_NAME ${DEPENDENT_OPTIONS})
        if(DEFINED ${OPTION_NAME})
            # Check if this option was manually changed by comparing with expected value
            set(EXPECTED_VALUE ${LLAMALIB_RUNTIME_DETECTION_PREV})
            if(LLAMALIB_RUNTIME_DETECTION_PREV STREQUAL "")
                set(EXPECTED_VALUE ON)  # Default value
            endif()
            
            # If option differs from expected value, user manually changed it
            if(NOT "${${OPTION_NAME}}" STREQUAL "${EXPECTED_VALUE}" AND NOT ${OPTION_NAME}_MANUAL_OVERRIDE)
                set(${OPTION_NAME}_MANUAL_OVERRIDE TRUE CACHE INTERNAL "Track if ${OPTION_NAME} was manually set by user" FORCE)
            endif()
            
            # Update dependent options based on LLAMALIB_RUNTIME_DETECTION
            if(RUNTIME_DETECTION_CHANGED OR NOT DEFINED ${OPTION_NAME}_LAST_AUTO_VALUE)
                if(NOT ${OPTION_NAME}_MANUAL_OVERRIDE)
                    set(NEW_VALUE ${LLAMALIB_RUNTIME_DETECTION})
                    if(NOT "${${OPTION_NAME}}" STREQUAL "${NEW_VALUE}")
                        set(${OPTION_NAME} ${NEW_VALUE} CACHE BOOL "${description}" FORCE)
                        set(VALUE_AUTO_CHANGED ON)
                    endif()
                endif()
                set(${OPTION_NAME}_LAST_AUTO_VALUE ${${OPTION_NAME}} CACHE INTERNAL "Last automatically set value")
            endif()
        endif()
    endforeach()
    
    # Check for multiple enabled options when runtime detection is OFF
    if(NOT RUNTIME_DETECTION_CHANGED OR NOT VALUE_AUTO_CHANGED)
    # if(NOT VALUE_AUTO_CHANGED)
        set(ENABLED_OPTIONS "")
        set(ENABLED_COUNT 0)
        
        foreach(OPTION_NAME ${DEPENDENT_OPTIONS})
            if(DEFINED ${OPTION_NAME} AND ${OPTION_NAME})
                list(APPEND ENABLED_OPTIONS ${OPTION_NAME})
                math(EXPR ENABLED_COUNT "${ENABLED_COUNT} + 1")
            endif()
        endforeach()
        
        if(ENABLED_COUNT EQUAL 0)
            set(MESSAGE_AT_LEAST "one")
            if(LLAMALIB_RUNTIME_DETECTION)
              set(MESSAGE_AT_LEAST "at least one")
            endif()
            message(WARNING "No architecture specified. Select ${MESSAGE_AT_LEAST} of the architectures (LLAMALIB_USE_*).")
        elseif(ENABLED_COUNT GREATER 1 AND NOT LLAMALIB_RUNTIME_DETECTION)
            message(WARNING "Select only one of the architectures (LLAMALIB_USE_*). You can select multiple only if LLAMALIB_RUNTIME_DETECTION is set.")
        endif()
    endif()

    # Update the previous state
    set(LLAMALIB_RUNTIME_DETECTION_PREV ${LLAMALIB_RUNTIME_DETECTION} CACHE INTERNAL "Previous state of LLAMALIB_RUNTIME_DETECTION" FORCE)
endfunction()

include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibCommon.cmake")

# Set default options
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(NOT DEFINED LLAMALIB_RUNTIME_DETECTION)
      OPTION(LLAMALIB_RUNTIME_DETECTION "Enable runtime detection" ON)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  option(LLAMALIB_ALLOW_CPU "Enable CPU support" ON)
  option(LLAMALIB_ALLOW_GPU "Enable GPU support" ON)

  if(LLAMALIB_ALLOW_GPU)
    llamalib_option(LLAMALIB_USE_CUBLAS "Enable CUBLAS architecture (GPU)")
    llamalib_option(LLAMALIB_USE_TINYBLAS "Enable tinyBLAS architecture (GPU)")
    llamalib_option(LLAMALIB_USE_VULKAN "Enable Vulkan architecture (GPU)")
    llamalib_option(LLAMALIB_USE_HIP "Enable HIP architecture (GPU)")
  else()
    foreach(opt LLAMALIB_USE_CUBLAS LLAMALIB_USE_TINYBLAS LLAMALIB_USE_VULKAN LLAMALIB_USE_HIP)
      set(${opt} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_GPU is OFF" FORCE)
    endforeach()
  endif()

  if(LLAMALIB_ALLOW_CPU)
    llamalib_option(LLAMALIB_USE_AVX512 "Enable AVX512 architecture")
    llamalib_option(LLAMALIB_USE_AVX2 "Enable AVX2 architecture")
    llamalib_option(LLAMALIB_USE_AVX "Enable AVX architecture")
    llamalib_option(LLAMALIB_USE_NOAVX "Enable no-AVX architecture")
  else()
    foreach(opt LLAMALIB_USE_AVX512 LLAMALIB_USE_AVX2 LLAMALIB_USE_AVX LLAMALIB_USE_NOAVX)
      set(${opt} OFF CACHE BOOL "Forced OFF because LLAMALIB_ALLOW_CPU is OFF" FORCE)
    endforeach()
  endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  llamalib_option(LLAMALIB_USE_ACCELERATE "Enable Accelerate framework")
  llamalib_option(LLAMALIB_USE_NO_ACCELERATE "Disable Accelerate framework")
endif()

if(NOT DEFINED LLAMALIB_COPY_DEPS)
    OPTION(LLAMALIB_COPY_DEPS "Automatically copy LlamaLib libraries" ON)
endif()

# Handle dependent options after all options are defined
handle_dependent_options()

# Include the main logic from a separate module
include("${CMAKE_CURRENT_LIST_DIR}/LlamaLibTargets.cmake")