#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerExceptionStatusMappingGate final
{
    DebugListenerExceptionStatusMappingGate() noexcept = default;
    DebugListenerExceptionStatusMappingGate(const DebugListenerExceptionStatusMappingGate&) = delete;
    DebugListenerExceptionStatusMappingGate& operator=(const DebugListenerExceptionStatusMappingGate&) = delete;
    DebugListenerExceptionStatusMappingGate(DebugListenerExceptionStatusMappingGate&&) = delete;
    DebugListenerExceptionStatusMappingGate& operator=(DebugListenerExceptionStatusMappingGate&&) = delete;
    ~DebugListenerExceptionStatusMappingGate() noexcept = default;

    int32_t map_exception_to_status(bool callback_failed) noexcept
    {
        exception_capture_ready = true;
        status_mapping_ready = true;
        last_status = callback_failed ? -1 : 0;
        if (callback_failed)
        {
            ++failure_count;
        }

        return last_status;
    }

    bool exception_escape_blocked() const noexcept
    {
        return true;
    }

    int32_t last_status{};
    int32_t failure_count{};
    bool exception_capture_ready{};
    bool status_mapping_ready{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerExceptionStatusMappingGate>::value,
    "debug-listener-exception-status-mapping-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerExceptionStatusMappingGate>::value,
    "debug-listener-exception-status-mapping-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerExceptionStatusMappingGate>::value,
    "debug-listener-exception-status-mapping-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerExceptionStatusMappingGate>::value,
    "debug-listener-exception-status-mapping-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerExceptionStatusMappingGate>::value,
    "debug-listener-exception-status-mapping-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerExceptionStatusMappingGate&>().map_exception_to_status(false)),
    "debug-listener-exception-status-mapping-gate map_exception_to_status must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerExceptionStatusMappingGate&>().exception_escape_blocked()),
    "debug-listener-exception-status-mapping-gate exception predicate must stay noexcept.");
} // namespace
