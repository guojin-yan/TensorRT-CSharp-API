#include "object.hpp"

#if defined(_MSC_VER)
#include <excpt.h>
#endif

#include <new>
#include <sstream>

#include "jyppx/common/runtime_info.h"
#include "../../common/error_state.hpp"

#ifndef JYPPX_HAS_TENSORRT
#define JYPPX_HAS_TENSORRT 0
#endif

#ifndef JYPPX_TENSORRT_VERSION_TEXT
#define JYPPX_TENSORRT_VERSION_TEXT "not-detected"
#endif

#ifndef JYPPX_TENSORRT_VERSION_MAJOR_NUM
#define JYPPX_TENSORRT_VERSION_MAJOR_NUM 0
#endif

namespace
{
const char* kind_to_name(const JYPPX_TensorRtObjectKind kind)
{
    switch (kind)
    {
    case JYPPX_TENSORRT_OBJECT_KIND_LOGGER:
        return "logger";
    case JYPPX_TENSORRT_OBJECT_KIND_RUNTIME:
        return "runtime";
    case JYPPX_TENSORRT_OBJECT_KIND_BUILDER:
        return "builder";
    case JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG:
        return "builder-config";
    case JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION:
        return "network-definition";
    case JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY:
        return "host-memory";
    case JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE:
        return "cuda-engine";
    case JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT:
        return "execution-context";
    case JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER:
        return "onnx-parser";
    case JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE:
        return "optimization-profile";
    case JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR:
        return "engine-inspector";
    case JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE:
        return "timing-cache";
    case JYPPX_TENSORRT_OBJECT_KIND_TENSOR:
        return "tensor";
    case JYPPX_TENSORRT_OBJECT_KIND_LAYER:
        return "layer";
    case JYPPX_TENSORRT_OBJECT_KIND_LOOP:
        return "loop";
    case JYPPX_TENSORRT_OBJECT_KIND_IF_CONDITIONAL:
        return "if-conditional";
    case JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG:
        return "serialization-config";
    case JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG:
        return "runtime-config";
    case JYPPX_TENSORRT_OBJECT_KIND_ATTENTION:
        return "attention";
    case JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR:
        return "progress-monitor";
    case JYPPX_TENSORRT_OBJECT_KIND_PROFILER:
        return "profiler";
    case JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER:
        return "onnx-parser-refitter";
    case JYPPX_TENSORRT_OBJECT_KIND_ALLOCATOR_CALLBACK_OWNER:
        return "allocator-callback-owner";
    case JYPPX_TENSORRT_OBJECT_KIND_ONNX_CONFIG:
        return "onnx-config";
    case JYPPX_TENSORRT_OBJECT_KIND_DEBUG_LISTENER_CALLBACK_OWNER:
        return "debug-listener-callback-owner";
    case JYPPX_TENSORRT_OBJECT_KIND_OUTPUT_ALLOCATOR_CALLBACK_OWNER:
        return "output-allocator-callback-owner";
    case JYPPX_TENSORRT_OBJECT_KIND_GPU_ALLOCATOR_CALLBACK_OWNER:
        return "gpu-allocator-callback-owner";
    default:
        return "unknown";
    }
}
}

namespace jyppx::tensorrt
{
JYPPX_StatusCode validate_output_pointer(void* pointer, const char* name)
{
    if (pointer != nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    std::ostringstream builder;
    builder << name << " output pointer must not be null.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_handle(const JYPPX_TensorRtObjectBase* object, const JYPPX_TensorRtLine expected_line, const JYPPX_TensorRtObjectKind expected_kind, const char* parameter_name)
{
    if (object == nullptr)
    {
        std::ostringstream builder;
        builder << parameter_name << " handle must not be null.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (object->magic != kObjectMagic)
    {
        std::ostringstream builder;
        builder << parameter_name << " does not point to a valid TensorRT bridge object.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (expected_line != JYPPX_TENSORRT_LINE_UNKNOWN && object->line != static_cast<uint32_t>(expected_line))
    {
        std::ostringstream builder;
        builder << parameter_name << " belongs to TensorRT line " << object->line << " but the API expects line " << static_cast<uint32_t>(expected_line) << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (expected_kind != JYPPX_TENSORRT_OBJECT_KIND_UNKNOWN && object->kind != static_cast<uint32_t>(expected_kind))
    {
        std::ostringstream builder;
        builder << parameter_name << " is not a " << kind_to_name(expected_kind) << " handle.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_TensorRtObjectBase* create_object(const JYPPX_TensorRtLine line, const JYPPX_TensorRtObjectKind kind)
{
    auto* object = new (std::nothrow) JYPPX_TensorRtObjectBase();
    if (object == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unable to allocate TensorRT bridge object.");
        return nullptr;
    }

    object->magic = kObjectMagic;
    object->line = static_cast<uint32_t>(line);
    object->kind = static_cast<uint32_t>(kind);
    object->payload = nullptr;
    object->destroy_payload = nullptr;
    return object;
}

void attach_payload(JYPPX_TensorRtObjectBase* object, void* payload, void (*destroy_payload)(void*))
{
    if (object == nullptr)
    {
        return;
    }

    object->payload = payload;
    object->destroy_payload = destroy_payload;
}

void* get_payload(const JYPPX_TensorRtObjectBase* object)
{
    return object != nullptr ? object->payload : nullptr;
}

const char* line_to_version_text(const JYPPX_TensorRtLine line)
{
    switch (line)
    {
    case JYPPX_TENSORRT_LINE_8:
        return "8.x";
    case JYPPX_TENSORRT_LINE_10:
        return "10.x";
    case JYPPX_TENSORRT_LINE_11:
        return "11.x";
    default:
        return "unknown";
    }
}

JYPPX_StatusCode report_vendor_missing(const JYPPX_TensorRtLine line, const char* feature_name)
{
    std::ostringstream builder;
    builder << "TensorRT " << line_to_version_text(line) << " " << feature_name << " is unavailable because TensorRT was not detected when the bridge was built.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}

JYPPX_StatusCode report_vendor_mismatch(const JYPPX_TensorRtLine requested_line, const int32_t detected_major, const char* feature_name)
{
    std::ostringstream builder;
    builder << "TensorRT " << line_to_version_text(requested_line) << " " << feature_name
            << " is unavailable because the bridge was built against TensorRT " << detected_major << ".x.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode report_vendor_exception(const JYPPX_TensorRtLine line, const char* feature_name, const char* exception_message)
{
    std::ostringstream builder;
    builder << "TensorRT " << line_to_version_text(line) << " " << feature_name << " raised a native exception";
    if (exception_message != nullptr && exception_message[0] != '\0')
    {
        builder << ": " << exception_message;
    }

    builder << ".";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_RUNTIME_ERROR;
}

JYPPX_StatusCode report_vendor_seh_exception(const JYPPX_TensorRtLine line, const char* feature_name, const uint32_t exception_code)
{
    std::ostringstream builder;
    builder << "TensorRT " << line_to_version_text(line) << " " << feature_name
            << " raised a structured exception with code " << exception_code << ".";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_RUNTIME_ERROR;
}

#if defined(_MSC_VER)
int capture_vendor_seh_exception_code(uint32_t* out_exception_code, const uint32_t exception_code)
{
    constexpr uint32_t kMsvcCppExceptionCode = 0xE06D7363U;
    if (exception_code == kMsvcCppExceptionCode)
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    if (out_exception_code != nullptr)
    {
        *out_exception_code = exception_code;
    }

    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

JYPPX_StatusCode report_not_implemented(const JYPPX_TensorRtLine line, const char* feature_name)
{
    std::ostringstream builder;
    builder << "TensorRT " << line_to_version_text(line) << " " << feature_name << " is declared by this adapter but not wired to vendor objects yet.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

void fill_adapter_info(JYPPX_TensorRtAdapterInfo* out_info, const JYPPX_TensorRtLine line)
{
    out_info->line = static_cast<uint32_t>(line);
    out_info->vendor_dependency_available = JYPPX_HAS_TENSORRT ? JYPPX_TRUE : JYPPX_FALSE;
    const bool line_matches = (line == JYPPX_TENSORRT_LINE_10 && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10)
        || (line == JYPPX_TENSORRT_LINE_8 && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8)
        || (line == JYPPX_TENSORRT_LINE_11 && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11);
    out_info->runtime_creation_supported = (JYPPX_HAS_TENSORRT && line_matches) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->builder_creation_supported = (JYPPX_HAS_TENSORRT && line_matches) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->network_creation_supported = (JYPPX_HAS_TENSORRT && line_matches) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->engine_deserialization_supported = (JYPPX_HAS_TENSORRT && line_matches) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->detected_version = JYPPX_TENSORRT_VERSION_TEXT;
    if (!JYPPX_HAS_TENSORRT)
    {
        out_info->status_message = "TensorRT vendor dependency was not detected when the bridge was built.";
    }
    else if (!line_matches)
    {
        out_info->status_message = "TensorRT vendor dependency was detected, but it does not match this adapter line.";
    }
    else
    {
        out_info->status_message = "TensorRT vendor dependency is available for this adapter line.";
    }
}
}

JYPPX_StatusCode jyppx_trt_object_destroy(JYPPX_TensorRtObjectBase* object)
{
    if (object == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    if (object->magic != jyppx::tensorrt::kObjectMagic)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Attempted to destroy an invalid TensorRT bridge object.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (object->destroy_payload != nullptr && object->payload != nullptr)
    {
        object->destroy_payload(object->payload);
    }

    delete object;
    return JYPPX_STATUS_OK;
}
