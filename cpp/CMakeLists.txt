set (CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUML VERSION 0.15.0 LANGUAGES C CXX CUDA)

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

if(CMAKE_CUDA_HOST_COMPILER)
  # If CMAKE_CUDA_HOST_COMPILER is set to a nonempty string cmake was called with the environment variable CUDAHOSTCXX set or -DCMAKE_CUDA_HOST_COMPILER
  if(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
    message(WARNING "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER} and CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} are inconsistent!")
  endif(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
else()
  # No attempt to set CMAKE_CUDA_HOST_COMPILER has been made. Make sure CMAKE_CXX_COMPILER is used as CUDA host compiler.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
endif(CMAKE_CUDA_HOST_COMPILER)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14")
endif(${CMAKE_VERSION} VERSION_LESS "3.17.0")

include(cmake/EvalGpuArchs.cmake)
evaluate_gpu_archs(GPU_ARCHS)

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")

if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX11_ABI)
    message(STATUS "Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
  elseif(CMAKE_CXX11_ABI)
    message(STATUS "Enabling the GLIBCXX11 ABI")
  endif(NOT CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS
  "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=unrecognized_gcc_pragma")

##############################################################################
# - include paths ------------------------------------------------------------

set(CUML_INCLUDE_DIRECTORIES
    ${LEARNING_CUDA_INCLUDE_DIR})

##############################################################################
# - build count ------------------------------------------------------------
add_executable(count
    count/main.cu)

target_include_directories(count PRIVATE ${LEARNING_CUDA_INCLUDE_DIR})
