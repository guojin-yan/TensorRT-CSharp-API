#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeOwnerVTableInstallExperiment final
{
    DebugListenerNativeOwnerVTableInstallExperiment() noexcept = default;
    DebugListenerNativeOwnerVTableInstallExperiment(const DebugListenerNativeOwnerVTableInstallExperiment&) = delete;
    DebugListenerNativeOwnerVTableInstallExperiment& operator=(const DebugListenerNativeOwnerVTableInstallExperiment&) = delete;
    DebugListenerNativeOwnerVTableInstallExperiment(DebugListenerNativeOwnerVTableInstallExperiment&&) = delete;
    DebugListenerNativeOwnerVTableInstallExperiment& operator=(DebugListenerNativeOwnerVTableInstallExperiment&&) = delete;
    ~DebugListenerNativeOwnerVTableInstallExperiment() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool configure_prerequisites(
        bool lifecycle_ready,
        bool attach_bridge_ready,
        bool no_throw_vtable_ready,
        bool metadata_gate_ready,
        bool install_preflight_ready) noexcept
    {
        owner_lifecycle_gate_ready = lifecycle_ready;
        attach_bridge_shape_gate_ready = attach_bridge_ready;
        no_throw_vtable_scaffold_gate_ready = no_throw_vtable_ready;
        borrowed_metadata_gate_ready = metadata_gate_ready;
        native_vtable_install_preflight_ready = install_preflight_ready;
        return experiment_shape_ready();
    }

    bool configure_disabled_reason(bool non_null_attach_enabled_value, bool runtime_proof_enabled_value) noexcept
    {
        non_null_attach_enabled_value_seen = non_null_attach_enabled_value;
        runtime_proof_enabled_value_seen = runtime_proof_enabled_value;
        return install_attempt_guard_ready();
    }

    bool experiment_shape_ready() const noexcept
    {
        return line_supports_debug_listener &&
            owner_lifecycle_gate_ready &&
            attach_bridge_shape_gate_ready &&
            no_throw_vtable_scaffold_gate_ready &&
            borrowed_metadata_gate_ready &&
            native_vtable_install_preflight_ready;
    }

    bool install_attempt_guard_ready() const noexcept
    {
        return experiment_shape_ready() &&
            rollback_ready() &&
            detach_before_release_ready() &&
            failure_status_mapping_ready() &&
            pointer_free() &&
            !non_null_attach_enabled() &&
            !runtime_proof_enabled_value_seen &&
            !native_vtable_install_attempted() &&
            !native_vtable_installed();
    }

    bool non_null_attach_enabled() const noexcept
    {
        return false;
    }

    bool native_vtable_install_attempted() const noexcept
    {
        return false;
    }

    bool native_vtable_installed() const noexcept
    {
        return false;
    }

    bool rollback_ready() const noexcept
    {
        return true;
    }

    bool detach_before_release_ready() const noexcept
    {
        return true;
    }

    bool failure_status_mapping_ready() const noexcept
    {
        return true;
    }

    bool pointer_free() const noexcept
    {
        return true;
    }

    bool process_debug_tensor_runtime_ready() const noexcept
    {
        return false;
    }

    bool can_attempt_runtime_proof() const noexcept
    {
        return false;
    }

    bool line_supports_debug_listener{};
    bool owner_lifecycle_gate_ready{};
    bool attach_bridge_shape_gate_ready{};
    bool no_throw_vtable_scaffold_gate_ready{};
    bool borrowed_metadata_gate_ready{};
    bool native_vtable_install_preflight_ready{};
    bool non_null_attach_enabled_value_seen{};
    bool runtime_proof_enabled_value_seen{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeOwnerVTableInstallExperiment>::value,
    "debug-listener-native-owner-vtable-install-experiment copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeOwnerVTableInstallExperiment>::value,
    "debug-listener-native-owner-vtable-install-experiment copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeOwnerVTableInstallExperiment>::value,
    "debug-listener-native-owner-vtable-install-experiment move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeOwnerVTableInstallExperiment>::value,
    "debug-listener-native-owner-vtable-install-experiment move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeOwnerVTableInstallExperiment>::value,
    "debug-listener-native-owner-vtable-install-experiment destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeOwnerVTableInstallExperiment&>().configure_api_line(11)),
    "debug-listener-native-owner-vtable-install-experiment configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeOwnerVTableInstallExperiment&>().configure_prerequisites(true, true, true, true, true)),
    "debug-listener-native-owner-vtable-install-experiment configure_prerequisites must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeOwnerVTableInstallExperiment&>().configure_disabled_reason(false, false)),
    "debug-listener-native-owner-vtable-install-experiment configure_disabled_reason must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().experiment_shape_ready()),
    "debug-listener-native-owner-vtable-install-experiment shape predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().install_attempt_guard_ready()),
    "debug-listener-native-owner-vtable-install-experiment guard predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().non_null_attach_enabled()),
    "debug-listener-native-owner-vtable-install-experiment non-null attach predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().native_vtable_install_attempted()),
    "debug-listener-native-owner-vtable-install-experiment attempt predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().native_vtable_installed()),
    "debug-listener-native-owner-vtable-install-experiment installed predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().rollback_ready()),
    "debug-listener-native-owner-vtable-install-experiment rollback predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().detach_before_release_ready()),
    "debug-listener-native-owner-vtable-install-experiment detach predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().failure_status_mapping_ready()),
    "debug-listener-native-owner-vtable-install-experiment status predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().pointer_free()),
    "debug-listener-native-owner-vtable-install-experiment pointer-free predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().process_debug_tensor_runtime_ready()),
    "debug-listener-native-owner-vtable-install-experiment runtime predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerNativeOwnerVTableInstallExperiment&>().can_attempt_runtime_proof()),
    "debug-listener-native-owner-vtable-install-experiment proof predicate must stay noexcept.");
} // namespace
