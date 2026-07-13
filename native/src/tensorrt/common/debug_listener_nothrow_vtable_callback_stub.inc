#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNoThrowVTableCallbackStub final
{
    static constexpr int32_t max_rank = 8;
    static constexpr int32_t max_tensor_name = 128;

    DebugListenerNoThrowVTableCallbackStub() noexcept = default;
    DebugListenerNoThrowVTableCallbackStub(const DebugListenerNoThrowVTableCallbackStub&) = delete;
    DebugListenerNoThrowVTableCallbackStub& operator=(const DebugListenerNoThrowVTableCallbackStub&) = delete;
    DebugListenerNoThrowVTableCallbackStub(DebugListenerNoThrowVTableCallbackStub&&) = delete;
    DebugListenerNoThrowVTableCallbackStub& operator=(DebugListenerNoThrowVTableCallbackStub&&) = delete;
    ~DebugListenerNoThrowVTableCallbackStub() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    int32_t begin_callback(
        const char* tensor_name,
        int32_t data_type,
        int32_t tensor_location,
        int32_t rank,
        const int64_t* dims,
        bool is_input,
        bool is_execution_tensor) noexcept
    {
        ++callback_entry_count;
        ++inflight_callback_count;
        if (inflight_callback_count < 0)
        {
            inflight_never_negative = false;
            return complete_callback_failure(-3);
        }

        copy_tensor_name(tensor_name);
        copied_data_type = data_type;
        copied_tensor_location = tensor_location;
        copied_rank = clamp_rank(rank);
        for (int32_t index = 0; index < copied_rank; ++index)
        {
            copied_dims[index] = dims == nullptr ? 0 : dims[index];
        }

        for (int32_t index = copied_rank; index < max_rank; ++index)
        {
            copied_dims[index] = 0;
        }

        copied_is_input = is_input;
        copied_is_execution_tensor = is_execution_tensor;
        metadata_copy_ready = copied_name_length > 0 && copied_rank >= 0;
        return 0;
    }

    int32_t complete_callback_success() noexcept
    {
        last_status_code = 0;
        leave_callback();
        return last_status_code;
    }

    int32_t complete_callback_failure(int32_t status_code) noexcept
    {
        ++failure_count;
        last_status_code = status_code == 0 ? -1 : status_code;
        leave_callback();
        return last_status_code;
    }

    bool exception_escape_blocked() const noexcept
    {
        return true;
    }

    bool can_return_status_without_throwing() const noexcept
    {
        return true;
    }

    int32_t callback_entry_count{};
    int32_t callback_leave_count{};
    int32_t failure_count{};
    int32_t last_status_code{};
    int32_t inflight_callback_count{};
    bool metadata_copy_ready{};
    bool borrowed_pointer_escape_blocked{true};
    bool inflight_never_negative{true};
    bool line_supports_debug_listener{};
    char copied_tensor_name[max_tensor_name]{};
    int32_t copied_name_length{};
    int32_t copied_data_type{};
    int32_t copied_tensor_location{};
    int32_t copied_rank{};
    int64_t copied_dims[max_rank]{};
    bool copied_is_input{};
    bool copied_is_execution_tensor{};

private:
    static int32_t clamp_rank(int32_t rank) noexcept
    {
        if (rank < 0)
        {
            return 0;
        }

        return rank > max_rank ? max_rank : rank;
    }

    void copy_tensor_name(const char* tensor_name) noexcept
    {
        copied_name_length = 0;
        std::memset(copied_tensor_name, 0, sizeof(copied_tensor_name));
        if (tensor_name == nullptr)
        {
            return;
        }

        while (copied_name_length < max_tensor_name - 1 && tensor_name[copied_name_length] != '\0')
        {
            copied_tensor_name[copied_name_length] = tensor_name[copied_name_length];
            ++copied_name_length;
        }
    }

    void leave_callback() noexcept
    {
        if (inflight_callback_count <= 0)
        {
            inflight_never_negative = false;
            inflight_callback_count = 0;
        }
        else
        {
            --inflight_callback_count;
        }

        ++callback_leave_count;
    }
};

static_assert(
    !std::is_copy_constructible<DebugListenerNoThrowVTableCallbackStub>::value,
    "debug-listener-nothrow-vtable-callback-stub copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNoThrowVTableCallbackStub>::value,
    "debug-listener-nothrow-vtable-callback-stub copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNoThrowVTableCallbackStub>::value,
    "debug-listener-nothrow-vtable-callback-stub move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNoThrowVTableCallbackStub>::value,
    "debug-listener-nothrow-vtable-callback-stub move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNoThrowVTableCallbackStub>::value,
    "debug-listener-nothrow-vtable-callback-stub destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNoThrowVTableCallbackStub&>().configure_api_line(11)),
    "debug-listener-nothrow-vtable-callback-stub configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNoThrowVTableCallbackStub&>().begin_callback(nullptr, 0, 0, 0, nullptr, false, true)),
    "debug-listener-nothrow-vtable-callback-stub begin_callback must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNoThrowVTableCallbackStub&>().complete_callback_success()),
    "debug-listener-nothrow-vtable-callback-stub complete_callback_success must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNoThrowVTableCallbackStub&>().complete_callback_failure(-1)),
    "debug-listener-nothrow-vtable-callback-stub complete_callback_failure must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNoThrowVTableCallbackStub&>().exception_escape_blocked()),
    "debug-listener-nothrow-vtable-callback-stub exception predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNoThrowVTableCallbackStub&>().can_return_status_without_throwing()),
    "debug-listener-nothrow-vtable-callback-stub status predicate must stay noexcept.");
} // namespace
