#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerRealCallbackRuntimeProofReport final
{
    bool proof_gate_ready{};
    bool opt_in_enabled{};
    bool full_package_consumer_report{};
    bool runtime_smoke_ready{};
    bool trampoline_shape_ready{};
    bool attach_attempted{};
    bool attach_succeeded{};
    bool detach_attempted{};
    bool detach_succeeded{};
    bool rollback_attempted{};
    bool rollback_succeeded{};
    bool native_vtable_installed{};
    bool process_debug_tensor_invoked{};
    bool borrowed_debug_tensor_metadata_copied{};
    bool pointer_free_surface_ready{};
    bool process_debug_tensor_runtime_ready{};
    bool attempted_no_invocation{};
    bool can_promote_real_callback_runtime{};
    int32_t invocation_count{};
    int32_t failure_count{};
    int32_t in_flight_callback_count{};
    int32_t last_status{};
};

struct DebugListenerRealCallbackRuntimeProofGate final
{
    DebugListenerRealCallbackRuntimeProofGate() noexcept = default;
    DebugListenerRealCallbackRuntimeProofGate(const DebugListenerRealCallbackRuntimeProofGate&) = delete;
    DebugListenerRealCallbackRuntimeProofGate& operator=(const DebugListenerRealCallbackRuntimeProofGate&) = delete;
    DebugListenerRealCallbackRuntimeProofGate(DebugListenerRealCallbackRuntimeProofGate&&) = delete;
    DebugListenerRealCallbackRuntimeProofGate& operator=(DebugListenerRealCallbackRuntimeProofGate&&) = delete;
    ~DebugListenerRealCallbackRuntimeProofGate() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool configure_prerequisites(
        bool opt_in,
        bool full_package_report,
        bool runtime_smoke,
        bool trampoline_shape,
        bool metadata_copied,
        bool pointer_free) noexcept
    {
        opt_in_enabled = opt_in;
        full_package_consumer_report = full_package_report;
        runtime_smoke_ready = runtime_smoke;
        trampoline_shape_ready = trampoline_shape;
        borrowed_debug_tensor_metadata_copied = metadata_copied;
        pointer_free_surface_ready = pointer_free;
        return can_attempt_runtime_proof();
    }

    bool configure_attempt(
        bool attach_was_attempted,
        bool attach_was_successful,
        bool detach_was_attempted,
        bool detach_was_successful,
        bool rollback_was_attempted,
        bool rollback_was_successful,
        bool vtable_was_installed) noexcept
    {
        attach_attempted = attach_was_attempted;
        attach_succeeded = attach_was_successful;
        detach_attempted = detach_was_attempted;
        detach_succeeded = detach_was_successful;
        rollback_attempted = rollback_was_attempted;
        rollback_succeeded = rollback_was_successful;
        native_vtable_installed = vtable_was_installed;
        return attach_succeeded && detach_succeeded && (!rollback_attempted || rollback_succeeded);
    }

    bool configure_invocation(
        bool process_invoked,
        int32_t invocation_count_value,
        int32_t failure_count_value,
        int32_t in_flight_callback_count_value,
        bool process_runtime_ready,
        int32_t status) noexcept
    {
        process_debug_tensor_invoked = process_invoked;
        invocation_count = invocation_count_value;
        failure_count = failure_count_value;
        in_flight_callback_count = in_flight_callback_count_value;
        process_debug_tensor_runtime_ready = process_runtime_ready;
        last_status = status;
        attempted_no_invocation = opt_in_enabled && attach_attempted && !process_debug_tensor_invoked && invocation_count == 0;
        return can_promote_real_callback_runtime();
    }

    bool can_attempt_runtime_proof() const noexcept
    {
        return line_supports_debug_listener &&
            opt_in_enabled &&
            runtime_smoke_ready &&
            trampoline_shape_ready &&
            borrowed_debug_tensor_metadata_copied &&
            pointer_free_surface_ready;
    }

    bool can_promote_real_callback_runtime() const noexcept
    {
        return can_attempt_runtime_proof() &&
            full_package_consumer_report &&
            attach_succeeded &&
            detach_succeeded &&
            (!rollback_attempted || rollback_succeeded) &&
            native_vtable_installed &&
            process_debug_tensor_invoked &&
            invocation_count > 0 &&
            failure_count == 0 &&
            in_flight_callback_count == 0 &&
            process_debug_tensor_runtime_ready;
    }

    DebugListenerRealCallbackRuntimeProofReport make_report() const noexcept
    {
        DebugListenerRealCallbackRuntimeProofReport report{};
        report.proof_gate_ready = can_attempt_runtime_proof();
        report.opt_in_enabled = opt_in_enabled;
        report.full_package_consumer_report = full_package_consumer_report;
        report.runtime_smoke_ready = runtime_smoke_ready;
        report.trampoline_shape_ready = trampoline_shape_ready;
        report.attach_attempted = attach_attempted;
        report.attach_succeeded = attach_succeeded;
        report.detach_attempted = detach_attempted;
        report.detach_succeeded = detach_succeeded;
        report.rollback_attempted = rollback_attempted;
        report.rollback_succeeded = rollback_succeeded;
        report.native_vtable_installed = native_vtable_installed;
        report.process_debug_tensor_invoked = process_debug_tensor_invoked;
        report.borrowed_debug_tensor_metadata_copied = borrowed_debug_tensor_metadata_copied;
        report.pointer_free_surface_ready = pointer_free_surface_ready;
        report.process_debug_tensor_runtime_ready = process_debug_tensor_runtime_ready;
        report.attempted_no_invocation = attempted_no_invocation;
        report.can_promote_real_callback_runtime = can_promote_real_callback_runtime();
        report.invocation_count = invocation_count;
        report.failure_count = failure_count;
        report.in_flight_callback_count = in_flight_callback_count;
        report.last_status = last_status;
        return report;
    }

    bool line_supports_debug_listener{};
    bool opt_in_enabled{};
    bool full_package_consumer_report{};
    bool runtime_smoke_ready{};
    bool trampoline_shape_ready{};
    bool attach_attempted{};
    bool attach_succeeded{};
    bool detach_attempted{};
    bool detach_succeeded{};
    bool rollback_attempted{};
    bool rollback_succeeded{};
    bool native_vtable_installed{};
    bool process_debug_tensor_invoked{};
    bool borrowed_debug_tensor_metadata_copied{};
    bool pointer_free_surface_ready{};
    bool process_debug_tensor_runtime_ready{};
    bool attempted_no_invocation{};
    int32_t invocation_count{};
    int32_t failure_count{};
    int32_t in_flight_callback_count{};
    int32_t last_status{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerRealCallbackRuntimeProofGate>::value,
    "debug-listener-real-callback-runtime-proof copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerRealCallbackRuntimeProofGate>::value,
    "debug-listener-real-callback-runtime-proof copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerRealCallbackRuntimeProofGate>::value,
    "debug-listener-real-callback-runtime-proof move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerRealCallbackRuntimeProofGate>::value,
    "debug-listener-real-callback-runtime-proof move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerRealCallbackRuntimeProofGate>::value,
    "debug-listener-real-callback-runtime-proof destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealCallbackRuntimeProofGate&>().configure_api_line(11)),
    "debug-listener-real-callback-runtime-proof configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealCallbackRuntimeProofGate&>().configure_prerequisites(false, false, false, false, false, false)),
    "debug-listener-real-callback-runtime-proof configure_prerequisites must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealCallbackRuntimeProofGate&>().configure_attempt(false, false, false, false, false, false, false)),
    "debug-listener-real-callback-runtime-proof configure_attempt must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealCallbackRuntimeProofGate&>().configure_invocation(false, 0, 0, 0, false, 0)),
    "debug-listener-real-callback-runtime-proof configure_invocation must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealCallbackRuntimeProofGate&>().can_attempt_runtime_proof()),
    "debug-listener-real-callback-runtime-proof can_attempt_runtime_proof must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealCallbackRuntimeProofGate&>().can_promote_real_callback_runtime()),
    "debug-listener-real-callback-runtime-proof can_promote_real_callback_runtime must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealCallbackRuntimeProofGate&>().make_report()),
    "debug-listener-real-callback-runtime-proof make_report must stay noexcept.");
} // namespace
