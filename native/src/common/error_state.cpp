#include "error_state.hpp"

#include <algorithm>
#include <cstring>

namespace
{
thread_local JYPPX_ErrorCategory g_last_error_category = JYPPX_ERROR_CATEGORY_NONE;
thread_local std::string g_last_error_message;
}

namespace jyppx::common
{
void set_last_error(JYPPX_ErrorCategory category, const char* message)
{
    g_last_error_category = category;
    g_last_error_message = message != nullptr ? message : "";
}

void clear_last_error()
{
    g_last_error_category = JYPPX_ERROR_CATEGORY_NONE;
    g_last_error_message.clear();
}

JYPPX_ErrorCategory get_last_error_category()
{
    return g_last_error_category;
}

const std::string& get_last_error_message()
{
    return g_last_error_message;
}

JYPPX_StatusCode copy_last_error_message(char* buffer, size_t buffer_size, size_t* out_required_size)
{
    const size_t required_size = g_last_error_message.size() + 1;
    if (out_required_size != nullptr)
    {
        *out_required_size = required_size;
    }

    if (buffer == nullptr)
    {
        return buffer_size == 0 ? JYPPX_STATUS_OK : JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (buffer_size < required_size)
    {
        if (buffer_size > 0)
        {
            const size_t copy_length = std::min(buffer_size - 1, g_last_error_message.size());
            if (copy_length > 0)
            {
                std::memcpy(buffer, g_last_error_message.data(), copy_length);
            }
            buffer[copy_length] = '\0';
        }

        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    if (required_size > 1)
    {
        std::memcpy(buffer, g_last_error_message.data(), g_last_error_message.size());
    }
    buffer[required_size - 1] = '\0';
    return JYPPX_STATUS_OK;
}
}

