#include <cstdint>
#include <type_traits>

namespace
{
struct DebugListenerNativeOwnerNonCopyableStorage final
{
    DebugListenerNativeOwnerNonCopyableStorage() noexcept = default;
    DebugListenerNativeOwnerNonCopyableStorage(const DebugListenerNativeOwnerNonCopyableStorage&) = delete;
    DebugListenerNativeOwnerNonCopyableStorage& operator=(const DebugListenerNativeOwnerNonCopyableStorage&) = delete;
    DebugListenerNativeOwnerNonCopyableStorage(DebugListenerNativeOwnerNonCopyableStorage&&) = delete;
    DebugListenerNativeOwnerNonCopyableStorage& operator=(DebugListenerNativeOwnerNonCopyableStorage&&) = delete;
    ~DebugListenerNativeOwnerNonCopyableStorage() noexcept = default;

    uint64_t owner_id{};
    int32_t attach_state{};
    bool release_requested{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeOwnerNonCopyableStorage>::value,
    "debug-listener-native-owner-noncopyable-storage copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeOwnerNonCopyableStorage>::value,
    "debug-listener-native-owner-noncopyable-storage copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeOwnerNonCopyableStorage>::value,
    "debug-listener-native-owner-noncopyable-storage move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeOwnerNonCopyableStorage>::value,
    "debug-listener-native-owner-noncopyable-storage move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeOwnerNonCopyableStorage>::value,
    "debug-listener-native-owner-noncopyable-storage scaffold destructor must stay noexcept.");
} // namespace
