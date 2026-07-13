#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeNoThrowVTableScaffoldGate final
{
    DebugListenerNativeNoThrowVTableScaffoldGate() noexcept = default;
    DebugListenerNativeNoThrowVTableScaffoldGate(const DebugListenerNativeNoThrowVTableScaffoldGate&) = delete;
    DebugListenerNativeNoThrowVTableScaffoldGate& operator=(const DebugListenerNativeNoThrowVTableScaffoldGate&) = delete;
    DebugListenerNativeNoThrowVTableScaffoldGate(DebugListenerNativeNoThrowVTableScaffoldGate&&) = delete;
    DebugListenerNativeNoThrowVTableScaffoldGate& operator=(DebugListenerNativeNoThrowVTableScaffoldGate&&) = delete;
    ~DebugListenerNativeNoThrowVTableScaffoldGate() noexcept = default;

    int32_t process_debug_tensor_stub() noexcept
    {
        ++callback_entry_count;
        ++callback_leave_count;
        return 0;
    }

    bool exception_escape_blocked() const noexcept
    {
        return true;
    }

    int32_t callback_entry_count{};
    int32_t callback_leave_count{};
    bool borrowed_pointer_escape_blocked{true};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeNoThrowVTableScaffoldGate>::value,
    "debug-listener-native-nothrow-vtable-scaffold-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeNoThrowVTableScaffoldGate>::value,
    "debug-listener-native-nothrow-vtable-scaffold-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeNoThrowVTableScaffoldGate>::value,
    "debug-listener-native-nothrow-vtable-scaffold-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeNoThrowVTableScaffoldGate>::value,
    "debug-listener-native-nothrow-vtable-scaffold-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeNoThrowVTableScaffoldGate>::value,
    "debug-listener-native-nothrow-vtable-scaffold-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeNoThrowVTableScaffoldGate&>().process_debug_tensor_stub()),
    "debug-listener-native-nothrow-vtable-scaffold-gate process_debug_tensor_stub must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeNoThrowVTableScaffoldGate&>().exception_escape_blocked()),
    "debug-listener-native-nothrow-vtable-scaffold-gate exception predicate must stay noexcept.");
} // namespace
