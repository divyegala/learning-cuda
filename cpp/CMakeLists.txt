set (CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(LEARNING_CUDA VERSION 0.1.0 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

##############################################################################
# - Conda environment detection ----------------------------------------------

# conda-build conda package building detection
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    message(STATUS "Detected conda-build call, CMAKE_PREFIX_PATH unchanged.")

# otherwise detecting conda environment
else("$ENV{CONDA_BUILD}" STREQUAL "1")

  if(DETECT_CONDA_ENV)

    if(DEFINED ENV{CONDA_PREFIX})
      set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")

      message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

      if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set (CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})

      endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

    else(DEFINED ENV{CONDA_PREFIX})
      message(STATUS "No Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

    endif(DEFINED ENV{CONDA_PREFIX})

  else(DETECT_CONDA_ENV)
    message(STATUS "No Conda detection being used")

  endif(DETECT_CONDA_ENV)

endif("$ENV{CONDA_BUILD}" STREQUAL "1")

##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)

set(LEARNING_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
  "Path to learning-cuda include directories")

##############################################################################
# - Compiler Options  --------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES
    70
    75)

##############################################################################
# - include paths ------------------------------------------------------------

set(LEARNING_CUDA_INCLUDE_DIRECTORIES
    ${LEARNING_CUDA_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

##############################################################################
# - build count ------------------------------------------------------------
add_executable(count
    src/count/main.cu)

target_include_directories(count PRIVATE ${LEARNING_CUDA_INCLUDE_DIRECTORIES})
