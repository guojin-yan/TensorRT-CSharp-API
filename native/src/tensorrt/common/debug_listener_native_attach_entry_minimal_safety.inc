#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeAttachEntryMinimalSafety final
{
    DebugListenerNativeAttachEntryMinimalSafety() noexcept = default;
    DebugListenerNativeAttachEntryMinimalSafety(const DebugListenerNativeAttachEntryMinimalSafety&) = delete;
    DebugListenerNativeAttachEntryMinimalSafety& operator=(const DebugListenerNativeAttachEntryMinimalSafety&) = delete;
    DebugListenerNativeAttachEntryMinimalSafety(DebugListenerNativeAttachEntryMinimalSafety&&) = delete;
    DebugListenerNativeAttachEntryMinimalSafety& operator=(DebugListenerNativeAttachEntryMinimalSafety&&) = delete;
    ~DebugListenerNativeAttachEntryMinimalSafety() noexcept = default;

    bool configure(int32_t api_line, bool runtime_scaffold_ready, bool lifecycle_gate_ready) noexcept
    {
        line = api_line;
        version_guard_ready = api_line == 10 || api_line == 11;
        parameter_shape_ready = version_guard_ready && runtime_scaffold_ready;
        no_throw_ready = parameter_shape_ready && lifecycle_gate_ready;
        ownership_diagnostics_ready = no_throw_ready;
        pointer_free = true;
        set_debug_listener_non_null_enabled = false;
        native_attach_would_be_blocked = true;
        process_debug_tensor_runtime_ready = false;
        source_visible_attach_entry_shape_located =
            parameter_shape_ready &&
            no_throw_ready &&
            ownership_diagnostics_ready &&
            pointer_free &&
            !set_debug_listener_non_null_enabled &&
            native_attach_would_be_blocked;
        return source_visible_attach_entry_shape_located;
    }

    bool can_call_set_debug_listener_non_null() const noexcept
    {
        return false;
    }

    int32_t line{};
    bool version_guard_ready{};
    bool parameter_shape_ready{};
    bool no_throw_ready{};
    bool ownership_diagnostics_ready{};
    bool pointer_free{true};
    bool set_debug_listener_non_null_enabled{};
    bool native_attach_would_be_blocked{true};
    bool process_debug_tensor_runtime_ready{};
    bool source_visible_attach_entry_shape_located{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeAttachEntryMinimalSafety>::value,
    "debug-listener-native-attach-entry-minimal-safety copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeAttachEntryMinimalSafety>::value,
    "debug-listener-native-attach-entry-minimal-safety copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeAttachEntryMinimalSafety>::value,
    "debug-listener-native-attach-entry-minimal-safety move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeAttachEntryMinimalSafety>::value,
    "debug-listener-native-attach-entry-minimal-safety move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeAttachEntryMinimalSafety>::value,
    "debug-listener-native-attach-entry-minimal-safety destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeAttachEntryMinimalSafety&>().configure(11, true, true)),
    "debug-listener-native-attach-entry-minimal-safety configure must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeAttachEntryMinimalSafety&>().can_call_set_debug_listener_non_null()),
    "debug-listener-native-attach-entry-minimal-safety non-null attach predicate must stay noexcept.");
} // namespace
