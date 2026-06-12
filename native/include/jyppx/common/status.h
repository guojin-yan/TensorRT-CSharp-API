#pragma once

#include <stdint.h>

#include "jyppx/common/bridge_exports.h"

typedef enum JYPPX_StatusCode
{
    JYPPX_STATUS_OK = 0,
    JYPPX_STATUS_INVALID_ARGUMENT = 1,
    JYPPX_STATUS_BUFFER_TOO_SMALL = 2,
    JYPPX_STATUS_NOT_FOUND = 3,
    JYPPX_STATUS_NOT_SUPPORTED = 4,
    JYPPX_STATUS_DEPENDENCY_MISSING = 5,
    JYPPX_STATUS_NOT_READY = 6,
    JYPPX_STATUS_RUNTIME_ERROR = 7,
    JYPPX_STATUS_INVALID_STATE = 8,
    JYPPX_STATUS_OUT_OF_MEMORY = 9,
    JYPPX_STATUS_NOT_IMPLEMENTED = 10
} JYPPX_StatusCode;

typedef enum JYPPX_ErrorCategory
{
    JYPPX_ERROR_CATEGORY_NONE = 0,
    JYPPX_ERROR_CATEGORY_COMMON = 1,
    JYPPX_ERROR_CATEGORY_CUDA = 2,
    JYPPX_ERROR_CATEGORY_TENSORRT = 3,
    JYPPX_ERROR_CATEGORY_IO = 4
} JYPPX_ErrorCategory;

typedef int32_t JYPPX_Boolean;

#define JYPPX_FALSE 0
#define JYPPX_TRUE 1

JYPPX_C_API(const char*) jyppx_common_status_code_to_string(JYPPX_StatusCode status_code);
JYPPX_C_API(const char*) jyppx_common_error_category_to_string(JYPPX_ErrorCategory category);

