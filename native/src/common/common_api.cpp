#include "jyppx/common/runtime_info.h"

#include "error_state.hpp"

namespace
{
const char* status_to_string(JYPPX_StatusCode status_code)
{
    switch (status_code)
    {
    case JYPPX_STATUS_OK:
        return "ok";
    case JYPPX_STATUS_INVALID_ARGUMENT:
        return "invalid-argument";
    case JYPPX_STATUS_BUFFER_TOO_SMALL:
        return "buffer-too-small";
    case JYPPX_STATUS_NOT_FOUND:
        return "not-found";
    case JYPPX_STATUS_NOT_SUPPORTED:
        return "not-supported";
    case JYPPX_STATUS_DEPENDENCY_MISSING:
        return "dependency-missing";
    case JYPPX_STATUS_NOT_READY:
        return "not-ready";
    case JYPPX_STATUS_RUNTIME_ERROR:
        return "runtime-error";
    case JYPPX_STATUS_INVALID_STATE:
        return "invalid-state";
    case JYPPX_STATUS_OUT_OF_MEMORY:
        return "out-of-memory";
    case JYPPX_STATUS_NOT_IMPLEMENTED:
        return "not-implemented";
    default:
        return "unknown-status";
    }
}

const char* category_to_string(JYPPX_ErrorCategory category)
{
    switch (category)
    {
    case JYPPX_ERROR_CATEGORY_NONE:
        return "none";
    case JYPPX_ERROR_CATEGORY_COMMON:
        return "common";
    case JYPPX_ERROR_CATEGORY_CUDA:
        return "cuda";
    case JYPPX_ERROR_CATEGORY_TENSORRT:
        return "tensorrt";
    case JYPPX_ERROR_CATEGORY_IO:
        return "io";
    default:
        return "unknown-category";
    }
}
}

const char* jyppx_common_status_code_to_string(JYPPX_StatusCode status_code)
{
    return status_to_string(status_code);
}

const char* jyppx_common_error_category_to_string(JYPPX_ErrorCategory category)
{
    return category_to_string(category);
}

JYPPX_StatusCode jyppx_common_get_last_error_message(char* buffer, size_t buffer_size, size_t* out_required_size)
{
    return jyppx::common::copy_last_error_message(buffer, buffer_size, out_required_size);
}

JYPPX_ErrorCategory jyppx_common_get_last_error_category(void)
{
    return jyppx::common::get_last_error_category();
}

void jyppx_common_clear_last_error(void)
{
    jyppx::common::clear_last_error();
}

