#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeOwnerLifecycleGate final
{
    enum class State : int32_t
    {
        detached = 0,
        attached = 1,
        detach_requested = 2,
        release_requested = 3,
        released = 4
    };

    DebugListenerNativeOwnerLifecycleGate() noexcept = default;
    DebugListenerNativeOwnerLifecycleGate(const DebugListenerNativeOwnerLifecycleGate&) = delete;
    DebugListenerNativeOwnerLifecycleGate& operator=(const DebugListenerNativeOwnerLifecycleGate&) = delete;
    DebugListenerNativeOwnerLifecycleGate(DebugListenerNativeOwnerLifecycleGate&&) = delete;
    DebugListenerNativeOwnerLifecycleGate& operator=(DebugListenerNativeOwnerLifecycleGate&&) = delete;
    ~DebugListenerNativeOwnerLifecycleGate() noexcept = default;

    void request_detach_before_release() noexcept
    {
        detach_requested = true;
        state = State::detach_requested;
    }

    void enter_callback() noexcept
    {
        ++in_flight_callback_count;
    }

    void leave_callback() noexcept
    {
        if (in_flight_callback_count > 0)
        {
            --in_flight_callback_count;
        }
    }

    bool can_unpin_after_detach() const noexcept
    {
        return detach_requested && release_requested && in_flight_callback_count == 0;
    }

    void request_release() noexcept
    {
        if (!release_requested)
        {
            release_requested = true;
            ++release_hook_count;
        }

        if (!detach_requested)
        {
            request_detach_before_release();
        }

        state = State::release_requested;
        if (can_unpin_after_detach())
        {
            callback_state_pinned = false;
            delegate_pinned = false;
            state = State::released;
        }
    }

    uint64_t owner_id{};
    int32_t release_hook_count{};
    int32_t in_flight_callback_count{};
    bool detach_requested{};
    bool release_requested{};
    bool callback_state_pinned{true};
    bool delegate_pinned{true};
    State state{State::detached};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeOwnerLifecycleGate>::value,
    "debug-listener-native-owner-lifecycle-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeOwnerLifecycleGate>::value,
    "debug-listener-native-owner-lifecycle-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeOwnerLifecycleGate>::value,
    "debug-listener-native-owner-lifecycle-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeOwnerLifecycleGate>::value,
    "debug-listener-native-owner-lifecycle-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeOwnerLifecycleGate>::value,
    "debug-listener-native-owner-lifecycle-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeOwnerLifecycleGate&>().request_detach_before_release()),
    "debug-listener-native-owner-lifecycle-gate detach request must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeOwnerLifecycleGate&>().request_release()),
    "debug-listener-native-owner-lifecycle-gate release request must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerLifecycleGate&>().can_unpin_after_detach()),
    "debug-listener-native-owner-lifecycle-gate unpin predicate must stay noexcept.");
} // namespace
