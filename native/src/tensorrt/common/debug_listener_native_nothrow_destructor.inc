#include <cstdint>
#include <type_traits>

namespace
{
struct DebugListenerNativeNoThrowDestructor final
{
    DebugListenerNativeNoThrowDestructor() noexcept = default;
    DebugListenerNativeNoThrowDestructor(const DebugListenerNativeNoThrowDestructor&) = delete;
    DebugListenerNativeNoThrowDestructor& operator=(const DebugListenerNativeNoThrowDestructor&) = delete;
    DebugListenerNativeNoThrowDestructor(DebugListenerNativeNoThrowDestructor&&) = delete;
    DebugListenerNativeNoThrowDestructor& operator=(DebugListenerNativeNoThrowDestructor&&) = delete;
    ~DebugListenerNativeNoThrowDestructor() noexcept = default;

    uint64_t owner_id{};
    int32_t attach_state{};
    bool detach_requested{};
    bool release_requested{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeNoThrowDestructor>::value,
    "debug-listener-native-nothrow-destructor copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeNoThrowDestructor>::value,
    "debug-listener-native-nothrow-destructor copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeNoThrowDestructor>::value,
    "debug-listener-native-nothrow-destructor move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeNoThrowDestructor>::value,
    "debug-listener-native-nothrow-destructor move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeNoThrowDestructor>::value,
    "debug-listener-native-nothrow-destructor scaffold destructor must stay noexcept.");
} // namespace
