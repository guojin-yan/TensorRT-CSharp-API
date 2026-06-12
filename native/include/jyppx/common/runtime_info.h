#pragma once

#include <stddef.h>
#include <stdint.h>

#include "jyppx/common/bridge_exports.h"
#include "jyppx/common/status.h"

typedef struct JYPPX_BuildInfo
{
    uint32_t abi_version;
    uint32_t bridge_version_major;
    uint32_t bridge_version_minor;
    uint32_t bridge_version_patch;
    const char* bridge_name;
    const char* bridge_banner;
    const char* compiler_id;
    const char* compiler_version;
    const char* system_name;
    const char* system_processor;
    const char* build_configuration;
    const char* cuda_toolkit_version;
    const char* tensorrt_version;
    JYPPX_Boolean has_cuda_toolkit;
    JYPPX_Boolean has_tensorrt;
    JYPPX_Boolean cuda_bindings_enabled;
    JYPPX_Boolean tensorrt_bindings_enabled;
} JYPPX_BuildInfo;

typedef struct JYPPX_RuntimeInfo
{
    uint32_t abi_version;
    const char* bridge_name;
    const char* bridge_banner;
    const char* last_error_message;
    JYPPX_ErrorCategory last_error_category;
    JYPPX_Boolean cuda_toolkit_available;
    JYPPX_Boolean tensorrt_available;
} JYPPX_RuntimeInfo;

typedef struct JYPPX_CapabilityInfo
{
    JYPPX_Boolean supports_trt8_adapter;
    JYPPX_Boolean supports_trt10_adapter;
    JYPPX_Boolean supports_trt11_adapter;
    JYPPX_Boolean supports_trt8_runtime_creation;
    JYPPX_Boolean supports_trt10_runtime_creation;
    JYPPX_Boolean supports_trt11_runtime_creation;
    JYPPX_Boolean supports_trt8_builder_creation;
    JYPPX_Boolean supports_trt10_builder_creation;
    JYPPX_Boolean supports_trt11_builder_creation;
    JYPPX_Boolean supports_last_error_query;
    JYPPX_Boolean supports_build_info_query;
    JYPPX_Boolean supports_runtime_info_query;
} JYPPX_CapabilityInfo;

JYPPX_C_API(JYPPX_StatusCode) jyppx_common_get_build_info(JYPPX_BuildInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_common_query_runtime_info(JYPPX_RuntimeInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_common_query_capability_info(JYPPX_CapabilityInfo* out_info);
JYPPX_C_API(JYPPX_StatusCode) jyppx_common_get_last_error_message(char* buffer, size_t buffer_size, size_t* out_required_size);
JYPPX_C_API(JYPPX_ErrorCategory) jyppx_common_get_last_error_category(void);
JYPPX_C_API(void) jyppx_common_clear_last_error(void);
