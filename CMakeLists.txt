cmake_minimum_required(VERSION 3.27)

project(TensorRtSharp VERSION 4.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/cmake")

option(JYPPX_ENABLE_TENSORRT_BINDINGS "Enable TensorRT vendor-aware bridge sources when dependencies are discovered." OFF)
option(JYPPX_ENABLE_CUDA_BINDINGS "Enable CUDA vendor-aware bridge sources when dependencies are discovered." OFF)
set(JYPPX_TARGET_ARCH "x64" CACHE STRING "Primary target architecture. Future values may include arm64.")
set_property(CACHE JYPPX_TARGET_ARCH PROPERTY STRINGS x64 arm64)
set(JYPPX_CUDA_VERSION "" CACHE STRING "Exact CUDA toolkit version, for example 11.8, 12.1, 12.9, or 13.2.")
set(JYPPX_TENSORRT_CUDA_VERSION "" CACHE STRING "CUDA version label used by the TensorRT package name when it differs from the CUDA toolkit line.")
set(JYPPX_CUDNN_MAJOR "" CACHE STRING "Preferred cuDNN major version for optional parser/runtime packaging checks.")

if(NOT JYPPX_TARGET_ARCH STREQUAL "x64")
  message(WARNING "This milestone is validated only for x64. Current value: ${JYPPX_TARGET_ARCH}")
endif()

set(JYPPX_OUTPUT_ROOT "${CMAKE_BINARY_DIR}")

function(jyppx_resolve_local_tensorrt_root out_var)
  if(DEFINED JYPPX_TENSORRT_ROOT AND EXISTS "${JYPPX_TENSORRT_ROOT}")
    set(${out_var} "${JYPPX_TENSORRT_ROOT}" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED TensorRT_ROOT AND EXISTS "${TensorRT_ROOT}")
    set(${out_var} "${TensorRT_ROOT}" PARENT_SCOPE)
    return()
  endif()

  if(NOT WIN32)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  if(NOT DEFINED JYPPX_TENSORRT_LINE OR NOT DEFINED JYPPX_CUDA_LINE)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  file(GLOB _jyppx_tensorrt_candidates LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/third_party/nvidia/TensorRT-*")
  set(_jyppx_selected "")
  foreach(_candidate ${_jyppx_tensorrt_candidates})
    get_filename_component(_candidate_name "${_candidate}" NAME)
    set(_jyppx_cuda_match FALSE)
    set(_jyppx_tensorrt_cuda_version "${JYPPX_CUDA_VERSION}")
    if(DEFINED JYPPX_TENSORRT_CUDA_VERSION AND NOT JYPPX_TENSORRT_CUDA_VERSION STREQUAL "")
      set(_jyppx_tensorrt_cuda_version "${JYPPX_TENSORRT_CUDA_VERSION}")
    endif()
    if(NOT _jyppx_tensorrt_cuda_version STREQUAL "")
      if(_candidate_name MATCHES "cuda ${_jyppx_tensorrt_cuda_version}(\\.|$)")
        set(_jyppx_cuda_match TRUE)
      endif()
    elseif(_candidate_name MATCHES "cuda ${JYPPX_CUDA_LINE}(\\.|$)")
      set(_jyppx_cuda_match TRUE)
    endif()

    if(_candidate_name MATCHES "^TensorRT-${JYPPX_TENSORRT_LINE}(\\.|-)" AND _jyppx_cuda_match)
      if(EXISTS "${_candidate}/include/NvInfer.h")
        set(_jyppx_selected "${_candidate}")
      endif()
    endif()
  endforeach()

  set(${out_var} "${_jyppx_selected}" PARENT_SCOPE)
endfunction()

function(jyppx_resolve_local_cuda_root out_var)
  if(DEFINED JYPPX_CUDA_ROOT AND EXISTS "${JYPPX_CUDA_ROOT}")
    set(${out_var} "${JYPPX_CUDA_ROOT}" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}")
    set(${out_var} "${CUDAToolkit_ROOT}" PARENT_SCOPE)
    return()
  endif()

  if(NOT WIN32)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  if(NOT DEFINED JYPPX_CUDA_LINE)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  set(_jyppx_cuda_base "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
  if(NOT EXISTS "${_jyppx_cuda_base}")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED JYPPX_CUDA_VERSION AND NOT JYPPX_CUDA_VERSION STREQUAL "")
    set(_jyppx_exact_cuda "${_jyppx_cuda_base}/v${JYPPX_CUDA_VERSION}")
    if(EXISTS "${_jyppx_exact_cuda}/include/cuda_runtime.h")
      set(${out_var} "${_jyppx_exact_cuda}" PARENT_SCOPE)
      return()
    endif()
  endif()

  file(GLOB _jyppx_cuda_candidates LIST_DIRECTORIES true "${_jyppx_cuda_base}/v${JYPPX_CUDA_LINE}.*")
  list(SORT _jyppx_cuda_candidates)

  if(DEFINED JYPPX_TENSORRT_ROOT)
    get_filename_component(_jyppx_trt_name "${JYPPX_TENSORRT_ROOT}" NAME)
  elseif(DEFINED TensorRT_ROOT)
    get_filename_component(_jyppx_trt_name "${TensorRT_ROOT}" NAME)
  else()
    set(_jyppx_trt_name "")
  endif()

  set(_jyppx_selected "")
  if(JYPPX_CUDA_LINE STREQUAL "12" AND _jyppx_trt_name MATCHES "^TensorRT-8\\." AND (_jyppx_trt_name MATCHES "cuda 12\\.0" OR _jyppx_trt_name MATCHES "cuda 12\\.1"))
    set(_jyppx_selected "${_jyppx_cuda_base}/v12.1")
  else()
    list(LENGTH _jyppx_cuda_candidates _jyppx_cuda_count)
    if(_jyppx_cuda_count GREATER 0)
      math(EXPR _jyppx_last_index "${_jyppx_cuda_count} - 1")
      list(GET _jyppx_cuda_candidates ${_jyppx_last_index} _jyppx_selected)
    endif()
  endif()

  set(${out_var} "${_jyppx_selected}" PARENT_SCOPE)
endfunction()

function(jyppx_resolve_local_cudnn_root out_var)
  if(DEFINED JYPPX_CUDNN_ROOT AND EXISTS "${JYPPX_CUDNN_ROOT}")
    if((JYPPX_CUDNN_MAJOR STREQUAL "9" AND EXISTS "${JYPPX_CUDNN_ROOT}/bin/cudnn64_9.dll") OR
       (JYPPX_CUDNN_MAJOR STREQUAL "8" AND EXISTS "${JYPPX_CUDNN_ROOT}/bin/cudnn64_8.dll") OR
       JYPPX_CUDNN_MAJOR STREQUAL "")
      set(${out_var} "${JYPPX_CUDNN_ROOT}" PARENT_SCOPE)
      return()
    endif()
  endif()

  if(NOT WIN32 OR NOT DEFINED JYPPX_CUDA_LINE)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  set(_jyppx_cudnn_patterns)
  if(JYPPX_CUDNN_MAJOR STREQUAL "9")
    list(APPEND _jyppx_cudnn_patterns "${CMAKE_SOURCE_DIR}/third_party/nvidia/cudnn_windows_x86_64_9.*_cuda${JYPPX_CUDA_LINE}")
  elseif(JYPPX_CUDNN_MAJOR STREQUAL "8")
    list(APPEND _jyppx_cudnn_patterns "${CMAKE_SOURCE_DIR}/third_party/nvidia/cudnn-windows-x86_64-8.*_cuda${JYPPX_CUDA_LINE}-archive")
  else()
    list(APPEND _jyppx_cudnn_patterns
      "${CMAKE_SOURCE_DIR}/third_party/nvidia/cudnn-windows-x86_64-8.*_cuda${JYPPX_CUDA_LINE}-archive"
      "${CMAKE_SOURCE_DIR}/third_party/nvidia/cudnn_windows_x86_64_9.*_cuda${JYPPX_CUDA_LINE}")
  endif()

  set(_jyppx_cudnn_candidates)
  foreach(_jyppx_cudnn_pattern ${_jyppx_cudnn_patterns})
    file(GLOB _jyppx_cudnn_matches LIST_DIRECTORIES true "${_jyppx_cudnn_pattern}")
    list(APPEND _jyppx_cudnn_candidates ${_jyppx_cudnn_matches})
  endforeach()
  list(SORT _jyppx_cudnn_candidates)
  foreach(_jyppx_selected_cudnn ${_jyppx_cudnn_candidates})
    if(EXISTS "${_jyppx_selected_cudnn}/bin/cudnn64_8.dll" OR EXISTS "${_jyppx_selected_cudnn}/bin/cudnn64_9.dll")
      set(${out_var} "${_jyppx_selected_cudnn}" PARENT_SCOPE)
      return()
    endif()

    file(GLOB _jyppx_nested_cudnn_dirs LIST_DIRECTORIES true "${_jyppx_selected_cudnn}/v*")
    list(SORT _jyppx_nested_cudnn_dirs)
    foreach(_jyppx_nested_cudnn ${_jyppx_nested_cudnn_dirs})
      if(EXISTS "${_jyppx_nested_cudnn}/bin/cudnn64_9.dll")
        set(${out_var} "${_jyppx_nested_cudnn}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()

  if(DEFINED CUDNN_ROOT AND EXISTS "${CUDNN_ROOT}")
    if((JYPPX_CUDNN_MAJOR STREQUAL "9" AND EXISTS "${CUDNN_ROOT}/bin/cudnn64_9.dll") OR
       (JYPPX_CUDNN_MAJOR STREQUAL "8" AND EXISTS "${CUDNN_ROOT}/bin/cudnn64_8.dll") OR
       JYPPX_CUDNN_MAJOR STREQUAL "")
      set(${out_var} "${CUDNN_ROOT}" PARENT_SCOPE)
      return()
    endif()
  endif()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

jyppx_resolve_local_tensorrt_root(_jyppx_resolved_tensorrt_root)
if(_jyppx_resolved_tensorrt_root)
  set(TensorRT_ROOT "${_jyppx_resolved_tensorrt_root}" CACHE PATH "Selected TensorRT package root" FORCE)
endif()

jyppx_resolve_local_cuda_root(_jyppx_resolved_cuda_root)
if(_jyppx_resolved_cuda_root)
  set(CUDAToolkit_ROOT "${_jyppx_resolved_cuda_root}" CACHE PATH "Selected CUDA toolkit root" FORCE)
  set(JYPPX_CUDA_ROOT "${_jyppx_resolved_cuda_root}" CACHE PATH "Selected CUDA package root" FORCE)
  set(CUDAToolkit_BIN_DIR "${_jyppx_resolved_cuda_root}/bin" CACHE PATH "Selected CUDA binary directory" FORCE)
  set(CUDAToolkit_INCLUDE_DIR "${_jyppx_resolved_cuda_root}/include" CACHE PATH "Selected CUDA include directory" FORCE)
  if(WIN32)
    set(_jyppx_selected_nvcc "${_jyppx_resolved_cuda_root}/bin/nvcc.exe")
  else()
    set(_jyppx_selected_nvcc "${_jyppx_resolved_cuda_root}/bin/nvcc")
  endif()
  if(EXISTS "${_jyppx_selected_nvcc}")
    set(CUDAToolkit_NVCC_EXECUTABLE "${_jyppx_selected_nvcc}" CACHE FILEPATH "Selected CUDA nvcc executable" FORCE)
  endif()
endif()

jyppx_resolve_local_cudnn_root(_jyppx_resolved_cudnn_root)
if(_jyppx_resolved_cudnn_root)
  set(JYPPX_CUDNN_ROOT "${_jyppx_resolved_cudnn_root}" CACHE PATH "Selected cuDNN package root" FORCE)
  set(CUDNN_ROOT "${_jyppx_resolved_cudnn_root}" CACHE PATH "Selected cuDNN package root" FORCE)
endif()

find_package(CUDAToolkit QUIET)
find_package(TensorRT QUIET)

function(jyppx_configure_local_tensorrt_fallback)
  if(TensorRT_FOUND)
    return()
  endif()

  if(NOT DEFINED TensorRT_ROOT OR NOT EXISTS "${TensorRT_ROOT}/include/NvInfer.h")
    return()
  endif()

  set(_jyppx_tensorrt_include "${TensorRT_ROOT}/include")
  set(_jyppx_tensorrt_lib_dirs "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64")

  set(_jyppx_nvinfer_names nvinfer)
  set(_jyppx_nvonnxparser_names nvonnxparser)
  if(WIN32 AND DEFINED JYPPX_TENSORRT_LINE AND NOT JYPPX_TENSORRT_LINE STREQUAL "8")
    list(INSERT _jyppx_nvinfer_names 0 "nvinfer_${JYPPX_TENSORRT_LINE}")
    list(INSERT _jyppx_nvonnxparser_names 0 "nvonnxparser_${JYPPX_TENSORRT_LINE}")
  endif()

  find_library(TensorRT_NVINFER_LIBRARY
    NAMES ${_jyppx_nvinfer_names}
    HINTS ${_jyppx_tensorrt_lib_dirs}
    NO_DEFAULT_PATH
  )
  if(NOT TensorRT_NVINFER_LIBRARY)
    return()
  endif()

  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED GLOBAL)
  endif()
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_jyppx_tensorrt_include}"
  )

  find_library(TensorRT_NVONNXPARSER_LIBRARY
    NAMES ${_jyppx_nvonnxparser_names}
    HINTS ${_jyppx_tensorrt_lib_dirs}
    NO_DEFAULT_PATH
  )
  if(TensorRT_NVONNXPARSER_LIBRARY AND NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED GLOBAL)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
      IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${_jyppx_tensorrt_include}"
    )
  endif()

  set(_jyppx_tensorrt_version_major "")
  set(_jyppx_tensorrt_version_minor "0")
  set(_jyppx_tensorrt_version_patch "0")
  get_filename_component(_jyppx_tensorrt_root_name "${TensorRT_ROOT}" NAME)
  if(_jyppx_tensorrt_root_name MATCHES "^TensorRT-([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(_jyppx_tensorrt_version_major "${CMAKE_MATCH_1}")
    set(_jyppx_tensorrt_version_minor "${CMAKE_MATCH_2}")
    set(_jyppx_tensorrt_version_patch "${CMAKE_MATCH_3}")
  elseif(DEFINED JYPPX_TENSORRT_LINE)
    set(_jyppx_tensorrt_version_major "${JYPPX_TENSORRT_LINE}")
  endif()

  if(NOT _jyppx_tensorrt_version_major STREQUAL "")
    set(TensorRT_VERSION_MAJOR "${_jyppx_tensorrt_version_major}" PARENT_SCOPE)
    set(TensorRT_VERSION_MINOR "${_jyppx_tensorrt_version_minor}" PARENT_SCOPE)
    set(TensorRT_VERSION_PATCH "${_jyppx_tensorrt_version_patch}" PARENT_SCOPE)
    set(TensorRT_VERSION "${_jyppx_tensorrt_version_major}.${_jyppx_tensorrt_version_minor}.${_jyppx_tensorrt_version_patch}" PARENT_SCOPE)
  endif()

  set(TensorRT_FOUND TRUE PARENT_SCOPE)
  set(TensorRT_INCLUDE_DIRS "${_jyppx_tensorrt_include}" PARENT_SCOPE)
  set(TensorRT_NVINFER_LIBRARY "${TensorRT_NVINFER_LIBRARY}" PARENT_SCOPE)
  set(TensorRT_NVONNXPARSER_LIBRARY "${TensorRT_NVONNXPARSER_LIBRARY}" PARENT_SCOPE)
endfunction()

jyppx_configure_local_tensorrt_fallback()

set(JYPPX_BRIDGE_VERSION_MAJOR 4)
set(JYPPX_BRIDGE_VERSION_MINOR 0)
set(JYPPX_BRIDGE_VERSION_PATCH 0)

set(JYPPX_HAS_CUDA_TOOLKIT_VALUE 0)
if(CUDAToolkit_FOUND)
  set(JYPPX_HAS_CUDA_TOOLKIT_VALUE 1)
endif()

set(JYPPX_HAS_TENSORRT_VALUE 0)
if(TensorRT_FOUND)
  set(JYPPX_HAS_TENSORRT_VALUE 1)
endif()

set(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE 0)
set(JYPPX_TENSORRT_ONNXPARSER_BLOCK_REASON "")

set(JYPPX_ENABLE_CUDA_BINDINGS_VALUE 0)
if(JYPPX_ENABLE_CUDA_BINDINGS)
  set(JYPPX_ENABLE_CUDA_BINDINGS_VALUE 1)
endif()

set(JYPPX_ENABLE_TENSORRT_BINDINGS_VALUE 0)
if(JYPPX_ENABLE_TENSORRT_BINDINGS)
  set(JYPPX_ENABLE_TENSORRT_BINDINGS_VALUE 1)
endif()

set(JYPPX_CUDA_TOOLKIT_VERSION_TEXT "not-detected")
if(CUDAToolkit_FOUND)
  set(JYPPX_CUDA_TOOLKIT_VERSION_TEXT "${CUDAToolkit_VERSION}")
endif()
if(_jyppx_resolved_cuda_root)
  get_filename_component(_jyppx_selected_cuda_name "${_jyppx_resolved_cuda_root}" NAME)
  if(_jyppx_selected_cuda_name MATCHES "^v[0-9]+\\.[0-9]+")
    string(REGEX REPLACE "^v" "" JYPPX_CUDA_TOOLKIT_VERSION_TEXT "${_jyppx_selected_cuda_name}")
  endif()
endif()

set(JYPPX_TENSORRT_VERSION_TEXT "not-detected")
if(TensorRT_FOUND AND TensorRT_VERSION)
  set(JYPPX_TENSORRT_VERSION_TEXT "${TensorRT_VERSION}")
endif()

set(JYPPX_TENSORRT_VERSION_MAJOR_NUM 0)
set(JYPPX_TENSORRT_VERSION_MINOR_NUM 0)
set(JYPPX_TENSORRT_VERSION_PATCH_NUM 0)
if(TensorRT_FOUND AND DEFINED TensorRT_VERSION_MAJOR)
  set(JYPPX_TENSORRT_VERSION_MAJOR_NUM "${TensorRT_VERSION_MAJOR}")
  set(JYPPX_TENSORRT_VERSION_MINOR_NUM "${TensorRT_VERSION_MINOR}")
  set(JYPPX_TENSORRT_VERSION_PATCH_NUM "${TensorRT_VERSION_PATCH}")
endif()

if(TensorRT_FOUND AND TensorRT_NVONNXPARSER_LIBRARY)
  set(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE 1)
  if(WIN32 AND JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 8)
    find_file(JYPPX_CUDNN8_RUNTIME_DLL
      NAMES cudnn64_8.dll
      HINTS
        "${CUDAToolkit_BIN_DIR}"
        "${CUDAToolkit_ROOT}/bin"
        "${JYPPX_CUDNN_ROOT}/bin"
        "${CUDNN_ROOT}/bin"
        "$ENV{CUDNN_ROOT}/bin"
        "$ENV{CUDNN_PATH}/bin"
      PATHS ENV PATH
    )
    if(NOT JYPPX_CUDNN8_RUNTIME_DLL)
      set(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE 0)
      set(JYPPX_TENSORRT_ONNXPARSER_BLOCK_REASON "TRT8 ONNX parser disabled because cudnn64_8.dll was not found.")
    endif()
  endif()
endif()

set(_jyppx_bridge_sources
  native/src/common/common_api.cpp
  native/src/common/error_state.cpp
  native/src/common/hello.cpp
  native/src/common/runtime_info.cpp
  native/src/cuda/api.cpp
  native/src/cuda/object.cpp
  native/src/tensorrt/common/object.cpp
)

if(TensorRT_FOUND AND JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 11)
  list(APPEND _jyppx_bridge_sources native/src/tensorrt/v11/api.cpp)
else()
  list(APPEND _jyppx_bridge_sources
    native/src/tensorrt/v8/api.cpp
    native/src/tensorrt/v10/api.cpp
  )
endif()

add_library(jyppxtrtbridge SHARED ${_jyppx_bridge_sources})

target_include_directories(jyppxtrtbridge
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/native/include"
)

set_target_properties(jyppxtrtbridge PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

if(CUDAToolkit_FOUND)
  target_include_directories(jyppxtrtbridge PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
  target_link_libraries(jyppxtrtbridge PRIVATE CUDA::cudart)
endif()

if(TensorRT_FOUND)
  target_include_directories(jyppxtrtbridge PRIVATE "${TensorRT_INCLUDE_DIRS}")
  target_link_libraries(jyppxtrtbridge PRIVATE TensorRT::nvinfer)
  if(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE AND TensorRT_NVONNXPARSER_LIBRARY AND TARGET TensorRT::nvonnxparser)
    target_link_libraries(jyppxtrtbridge PRIVATE TensorRT::nvonnxparser)
  endif()
endif()

set(_jyppx_windows_delay_load_dlls)
if(WIN32 AND MSVC)
  if(CUDAToolkit_FOUND)
    if(JYPPX_CUDA_TOOLKIT_VERSION_TEXT MATCHES "^11\\.")
      list(APPEND _jyppx_windows_delay_load_dlls "cudart64_110.dll")
    elseif(JYPPX_CUDA_TOOLKIT_VERSION_TEXT MATCHES "^12\\.")
      list(APPEND _jyppx_windows_delay_load_dlls "cudart64_12.dll")
    elseif(JYPPX_CUDA_TOOLKIT_VERSION_TEXT MATCHES "^13\\.")
      list(APPEND _jyppx_windows_delay_load_dlls "cudart64_13.dll")
    endif()
  endif()

  if(TensorRT_FOUND)
    if(JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 10)
      list(APPEND _jyppx_windows_delay_load_dlls "nvinfer_10.dll")
      if(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE)
        list(APPEND _jyppx_windows_delay_load_dlls "nvonnxparser_10.dll")
      endif()
    elseif(JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 11)
      list(APPEND _jyppx_windows_delay_load_dlls "nvinfer_11.dll")
      if(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE)
        list(APPEND _jyppx_windows_delay_load_dlls "nvonnxparser_11.dll")
      endif()
    elseif(JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 8)
      list(APPEND _jyppx_windows_delay_load_dlls "nvinfer.dll")
      if(JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE)
        list(APPEND _jyppx_windows_delay_load_dlls "nvonnxparser.dll")
      endif()
    endif()
  endif()

  if(_jyppx_windows_delay_load_dlls)
    list(REMOVE_DUPLICATES _jyppx_windows_delay_load_dlls)
    target_link_libraries(jyppxtrtbridge PRIVATE delayimp)
    foreach(_jyppx_delay_load_dll IN LISTS _jyppx_windows_delay_load_dlls)
      target_link_options(jyppxtrtbridge PRIVATE "/DELAYLOAD:${_jyppx_delay_load_dll}")
    endforeach()
  endif()
endif()

target_compile_definitions(jyppxtrtbridge
  PRIVATE
    JYPPX_BRIDGE_BUILD=1
    JYPPX_BRIDGE_VERSION_MAJOR=${JYPPX_BRIDGE_VERSION_MAJOR}
    JYPPX_BRIDGE_VERSION_MINOR=${JYPPX_BRIDGE_VERSION_MINOR}
    JYPPX_BRIDGE_VERSION_PATCH=${JYPPX_BRIDGE_VERSION_PATCH}
    JYPPX_BUILD_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    JYPPX_BUILD_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    JYPPX_BUILD_SYSTEM_NAME="${CMAKE_SYSTEM_NAME}"
    JYPPX_BUILD_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
    JYPPX_BUILD_CONFIGURATION="$<CONFIG>"
    JYPPX_HAS_CUDA_TOOLKIT=${JYPPX_HAS_CUDA_TOOLKIT_VALUE}
    JYPPX_HAS_TENSORRT=${JYPPX_HAS_TENSORRT_VALUE}
    JYPPX_HAS_TENSORRT_ONNXPARSER=${JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE}
    JYPPX_ENABLE_CUDA_BINDINGS_FLAG=${JYPPX_ENABLE_CUDA_BINDINGS_VALUE}
    JYPPX_ENABLE_TENSORRT_BINDINGS_FLAG=${JYPPX_ENABLE_TENSORRT_BINDINGS_VALUE}
    JYPPX_CUDA_TOOLKIT_VERSION_TEXT="${JYPPX_CUDA_TOOLKIT_VERSION_TEXT}"
    JYPPX_TENSORRT_VERSION_TEXT="${JYPPX_TENSORRT_VERSION_TEXT}"
    JYPPX_TENSORRT_VERSION_MAJOR_NUM=${JYPPX_TENSORRT_VERSION_MAJOR_NUM}
    JYPPX_TENSORRT_VERSION_MINOR_NUM=${JYPPX_TENSORRT_VERSION_MINOR_NUM}
    JYPPX_TENSORRT_VERSION_PATCH_NUM=${JYPPX_TENSORRT_VERSION_PATCH_NUM}
)

if(MSVC)
  target_compile_options(jyppxtrtbridge PRIVATE /W4 /permissive-)
else()
  target_compile_options(jyppxtrtbridge PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(jyppxtrtbridge
  PUBLIC
    JYPPX_BRIDGE_LOGICAL_NAME="jyppxtrtbridge"
)

message(STATUS "JYPPX_ENABLE_TENSORRT_BINDINGS=${JYPPX_ENABLE_TENSORRT_BINDINGS}")
message(STATUS "JYPPX_ENABLE_CUDA_BINDINGS=${JYPPX_ENABLE_CUDA_BINDINGS}")
message(STATUS "Selected TensorRT_ROOT=${TensorRT_ROOT}")
message(STATUS "Selected CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
message(STATUS "Selected CUDNN_ROOT=${CUDNN_ROOT}")
message(STATUS "TensorRT_FOUND=${TensorRT_FOUND}")
if(TensorRT_FOUND)
  message(STATUS "TensorRT_VERSION=${TensorRT_VERSION}")
endif()
message(STATUS "JYPPX_HAS_TENSORRT_ONNXPARSER=${JYPPX_HAS_TENSORRT_ONNXPARSER_VALUE}")
if(JYPPX_TENSORRT_ONNXPARSER_BLOCK_REASON)
  message(STATUS "${JYPPX_TENSORRT_ONNXPARSER_BLOCK_REASON}")
endif()
message(STATUS "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}")
if(CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")
endif()

if(TensorRT_FOUND AND JYPPX_TENSORRT_VERSION_MAJOR_NUM EQUAL 8)
  add_executable(jyppxtrt8probe
    native/tests/trt8_create_network_probe.cpp
  )

  target_include_directories(jyppxtrt8probe PRIVATE "${TensorRT_INCLUDE_DIRS}")
  if(CUDAToolkit_FOUND)
    target_include_directories(jyppxtrt8probe PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
  endif()
  target_link_libraries(jyppxtrt8probe PRIVATE TensorRT::nvinfer)

  set_target_properties(jyppxtrt8probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
  )
endif()
