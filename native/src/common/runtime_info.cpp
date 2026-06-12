#include "jyppx/common/hello.h"
#include "jyppx/common/runtime_info.h"

#include "error_state.hpp"

#ifndef JYPPX_BRIDGE_VERSION_MAJOR
#define JYPPX_BRIDGE_VERSION_MAJOR 0
#endif

#ifndef JYPPX_BRIDGE_VERSION_MINOR
#define JYPPX_BRIDGE_VERSION_MINOR 0
#endif

#ifndef JYPPX_BRIDGE_VERSION_PATCH
#define JYPPX_BRIDGE_VERSION_PATCH 0
#endif

#ifndef JYPPX_BUILD_COMPILER_ID
#define JYPPX_BUILD_COMPILER_ID "unknown"
#endif

#ifndef JYPPX_BUILD_COMPILER_VERSION
#define JYPPX_BUILD_COMPILER_VERSION "unknown"
#endif

#ifndef JYPPX_BUILD_SYSTEM_NAME
#define JYPPX_BUILD_SYSTEM_NAME "unknown"
#endif

#ifndef JYPPX_BUILD_SYSTEM_PROCESSOR
#define JYPPX_BUILD_SYSTEM_PROCESSOR "unknown"
#endif

#ifndef JYPPX_BUILD_CONFIGURATION
#define JYPPX_BUILD_CONFIGURATION "unknown"
#endif

#ifndef JYPPX_CUDA_TOOLKIT_VERSION_TEXT
#define JYPPX_CUDA_TOOLKIT_VERSION_TEXT "not-detected"
#endif

#ifndef JYPPX_TENSORRT_VERSION_TEXT
#define JYPPX_TENSORRT_VERSION_TEXT "not-detected"
#endif

#ifndef JYPPX_TENSORRT_VERSION_MAJOR_NUM
#define JYPPX_TENSORRT_VERSION_MAJOR_NUM 0
#endif

#ifndef JYPPX_HAS_CUDA_TOOLKIT
#define JYPPX_HAS_CUDA_TOOLKIT 0
#endif

#ifndef JYPPX_HAS_TENSORRT
#define JYPPX_HAS_TENSORRT 0
#endif

#ifndef JYPPX_ENABLE_CUDA_BINDINGS_FLAG
#define JYPPX_ENABLE_CUDA_BINDINGS_FLAG 0
#endif

#ifndef JYPPX_ENABLE_TENSORRT_BINDINGS_FLAG
#define JYPPX_ENABLE_TENSORRT_BINDINGS_FLAG 0
#endif

namespace
{
constexpr JYPPX_Boolean to_jyppx_bool(const int value)
{
    return value != 0 ? JYPPX_TRUE : JYPPX_FALSE;
}
}

JYPPX_StatusCode jyppx_common_get_build_info(JYPPX_BuildInfo* out_info)
{
    if (out_info == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_COMMON, "Build info output pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_info->abi_version = JYPPX_BRIDGE_ABI_VERSION;
    out_info->bridge_version_major = JYPPX_BRIDGE_VERSION_MAJOR;
    out_info->bridge_version_minor = JYPPX_BRIDGE_VERSION_MINOR;
    out_info->bridge_version_patch = JYPPX_BRIDGE_VERSION_PATCH;
    out_info->bridge_name = jyppx_common_get_bridge_name();
    out_info->bridge_banner = jyppx_common_get_bridge_banner();
    out_info->compiler_id = JYPPX_BUILD_COMPILER_ID;
    out_info->compiler_version = JYPPX_BUILD_COMPILER_VERSION;
    out_info->system_name = JYPPX_BUILD_SYSTEM_NAME;
    out_info->system_processor = JYPPX_BUILD_SYSTEM_PROCESSOR;
    out_info->build_configuration = JYPPX_BUILD_CONFIGURATION;
    out_info->cuda_toolkit_version = JYPPX_CUDA_TOOLKIT_VERSION_TEXT;
    out_info->tensorrt_version = JYPPX_TENSORRT_VERSION_TEXT;
    out_info->has_cuda_toolkit = to_jyppx_bool(JYPPX_HAS_CUDA_TOOLKIT);
    out_info->has_tensorrt = to_jyppx_bool(JYPPX_HAS_TENSORRT);
    out_info->cuda_bindings_enabled = to_jyppx_bool(JYPPX_ENABLE_CUDA_BINDINGS_FLAG);
    out_info->tensorrt_bindings_enabled = to_jyppx_bool(JYPPX_ENABLE_TENSORRT_BINDINGS_FLAG);

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_common_query_runtime_info(JYPPX_RuntimeInfo* out_info)
{
    if (out_info == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_COMMON, "Runtime info output pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_info->abi_version = JYPPX_BRIDGE_ABI_VERSION;
    out_info->bridge_name = jyppx_common_get_bridge_name();
    out_info->bridge_banner = jyppx_common_get_bridge_banner();
    out_info->last_error_message = jyppx::common::get_last_error_message().c_str();
    out_info->last_error_category = jyppx::common::get_last_error_category();
    out_info->cuda_toolkit_available = to_jyppx_bool(JYPPX_HAS_CUDA_TOOLKIT);
    out_info->tensorrt_available = to_jyppx_bool(JYPPX_HAS_TENSORRT);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_common_query_capability_info(JYPPX_CapabilityInfo* out_info)
{
    if (out_info == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_COMMON, "Capability info output pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_info->supports_trt8_adapter = JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11 ? JYPPX_FALSE : JYPPX_TRUE;
    out_info->supports_trt10_adapter = JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11 ? JYPPX_FALSE : JYPPX_TRUE;
    out_info->supports_trt11_adapter = JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11 ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt8_runtime_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt10_runtime_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt11_runtime_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt8_builder_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt10_builder_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_trt11_builder_creation = (JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->supports_last_error_query = JYPPX_TRUE;
    out_info->supports_build_info_query = JYPPX_TRUE;
    out_info->supports_runtime_info_query = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
}
