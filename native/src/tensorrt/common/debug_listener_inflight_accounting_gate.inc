#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerInFlightAccountingGate final
{
    DebugListenerInFlightAccountingGate() noexcept = default;
    DebugListenerInFlightAccountingGate(const DebugListenerInFlightAccountingGate&) = delete;
    DebugListenerInFlightAccountingGate& operator=(const DebugListenerInFlightAccountingGate&) = delete;
    DebugListenerInFlightAccountingGate(DebugListenerInFlightAccountingGate&&) = delete;
    DebugListenerInFlightAccountingGate& operator=(DebugListenerInFlightAccountingGate&&) = delete;
    ~DebugListenerInFlightAccountingGate() noexcept = default;

    void enter_callback() noexcept
    {
        ++in_flight_callback_count;
        if (in_flight_callback_count > max_in_flight_callback_count)
        {
            max_in_flight_callback_count = in_flight_callback_count;
        }
    }

    void leave_callback() noexcept
    {
        if (in_flight_callback_count > 0)
        {
            --in_flight_callback_count;
        }
    }

    bool can_release_after_drain() const noexcept
    {
        return in_flight_callback_count == 0;
    }

    int32_t in_flight_callback_count{};
    int32_t max_in_flight_callback_count{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerInFlightAccountingGate>::value,
    "debug-listener-inflight-accounting-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerInFlightAccountingGate>::value,
    "debug-listener-inflight-accounting-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerInFlightAccountingGate>::value,
    "debug-listener-inflight-accounting-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerInFlightAccountingGate>::value,
    "debug-listener-inflight-accounting-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerInFlightAccountingGate>::value,
    "debug-listener-inflight-accounting-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerInFlightAccountingGate&>().enter_callback()),
    "debug-listener-inflight-accounting-gate enter_callback must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerInFlightAccountingGate&>().leave_callback()),
    "debug-listener-inflight-accounting-gate leave_callback must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerInFlightAccountingGate&>().can_release_after_drain()),
    "debug-listener-inflight-accounting-gate release predicate must stay noexcept.");
} // namespace
