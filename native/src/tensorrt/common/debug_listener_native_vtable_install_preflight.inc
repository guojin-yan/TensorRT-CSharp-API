#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeVTableInstallPreflight final
{
    DebugListenerNativeVTableInstallPreflight() noexcept = default;
    DebugListenerNativeVTableInstallPreflight(const DebugListenerNativeVTableInstallPreflight&) = delete;
    DebugListenerNativeVTableInstallPreflight& operator=(const DebugListenerNativeVTableInstallPreflight&) = delete;
    DebugListenerNativeVTableInstallPreflight(DebugListenerNativeVTableInstallPreflight&&) = delete;
    DebugListenerNativeVTableInstallPreflight& operator=(DebugListenerNativeVTableInstallPreflight&&) = delete;
    ~DebugListenerNativeVTableInstallPreflight() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool configure_preflight(
        bool owner_lifecycle_ready,
        bool attach_bridge_shape_ready,
        bool no_throw_vtable_scaffold_ready,
        bool metadata_gate_ready) noexcept
    {
        owner_lifecycle_gate_ready = owner_lifecycle_ready;
        attach_bridge_shape_gate_ready = attach_bridge_shape_ready;
        native_nothrow_vtable_scaffold_gate_ready = no_throw_vtable_scaffold_ready;
        borrowed_debug_tensor_metadata_gate_ready = metadata_gate_ready;
        return preflight_shape_ready();
    }

    bool preflight_shape_ready() const noexcept
    {
        return owner_lifecycle_gate_ready &&
            attach_bridge_shape_gate_ready &&
            native_nothrow_vtable_scaffold_gate_ready &&
            borrowed_debug_tensor_metadata_gate_ready;
    }

    bool version_guard_ready() const noexcept
    {
        return line_supports_debug_listener;
    }

    bool no_throw_install_boundary_ready() const noexcept
    {
        return preflight_shape_ready();
    }

    bool ownership_diagnostics_ready() const noexcept
    {
        return preflight_shape_ready();
    }

    bool pointer_free() const noexcept
    {
        return true;
    }

    bool can_enable_set_debug_listener_non_null() const noexcept
    {
        return false;
    }

    bool can_install_native_vtable() const noexcept
    {
        return false;
    }

    bool native_vtable_installed() const noexcept
    {
        return false;
    }

    bool process_debug_tensor_runtime_ready() const noexcept
    {
        return false;
    }

    bool line_supports_debug_listener{};
    bool owner_lifecycle_gate_ready{};
    bool attach_bridge_shape_gate_ready{};
    bool native_nothrow_vtable_scaffold_gate_ready{};
    bool borrowed_debug_tensor_metadata_gate_ready{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeVTableInstallPreflight>::value,
    "debug-listener-native-vtable-install-preflight copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeVTableInstallPreflight>::value,
    "debug-listener-native-vtable-install-preflight copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeVTableInstallPreflight>::value,
    "debug-listener-native-vtable-install-preflight move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeVTableInstallPreflight>::value,
    "debug-listener-native-vtable-install-preflight move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeVTableInstallPreflight>::value,
    "debug-listener-native-vtable-install-preflight destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeVTableInstallPreflight&>().configure_api_line(11)),
    "debug-listener-native-vtable-install-preflight configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeVTableInstallPreflight&>().configure_preflight(true, true, true, true)),
    "debug-listener-native-vtable-install-preflight configure_preflight must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().preflight_shape_ready()),
    "debug-listener-native-vtable-install-preflight shape predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().version_guard_ready()),
    "debug-listener-native-vtable-install-preflight version predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().no_throw_install_boundary_ready()),
    "debug-listener-native-vtable-install-preflight no-throw predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().ownership_diagnostics_ready()),
    "debug-listener-native-vtable-install-preflight ownership predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().pointer_free()),
    "debug-listener-native-vtable-install-preflight pointer-free predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().can_enable_set_debug_listener_non_null()),
    "debug-listener-native-vtable-install-preflight non-null attach predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().can_install_native_vtable()),
    "debug-listener-native-vtable-install-preflight install predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().native_vtable_installed()),
    "debug-listener-native-vtable-install-preflight installed predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeVTableInstallPreflight&>().process_debug_tensor_runtime_ready()),
    "debug-listener-native-vtable-install-preflight runtime predicate must stay noexcept.");
} // namespace
