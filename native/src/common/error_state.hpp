#pragma once

#include <stddef.h>

#include <string>

#include "jyppx/common/status.h"

namespace jyppx::common
{
void set_last_error(JYPPX_ErrorCategory category, const char* message);
void clear_last_error();
JYPPX_ErrorCategory get_last_error_category();
const std::string& get_last_error_message();
JYPPX_StatusCode copy_last_error_message(char* buffer, size_t buffer_size, size_t* out_required_size);
}

