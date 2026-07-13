#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerRealNonNullAttachRuntimeSmokeAttempt final
{
    DebugListenerRealNonNullAttachRuntimeSmokeAttempt() noexcept = default;
    DebugListenerRealNonNullAttachRuntimeSmokeAttempt(const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&) = delete;
    DebugListenerRealNonNullAttachRuntimeSmokeAttempt& operator=(const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&) = delete;
    DebugListenerRealNonNullAttachRuntimeSmokeAttempt(DebugListenerRealNonNullAttachRuntimeSmokeAttempt&&) = delete;
    DebugListenerRealNonNullAttachRuntimeSmokeAttempt& operator=(DebugListenerRealNonNullAttachRuntimeSmokeAttempt&&) = delete;
    ~DebugListenerRealNonNullAttachRuntimeSmokeAttempt() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool configure_opt_in(bool enabled) noexcept
    {
        opt_in_enabled = enabled;
        return opt_in_enabled;
    }

    bool configure_prerequisites(
        bool attach_guard_ready_value,
        bool native_vtable_ready_value,
        bool borrowed_debug_tensor_runtime_ready_value,
        bool callback_invocation_ready_value,
        bool full_package_consumer_report_value) noexcept
    {
        attach_guard_ready_value_seen = attach_guard_ready_value;
        native_vtable_ready_value_seen = native_vtable_ready_value;
        borrowed_debug_tensor_runtime_ready_value_seen = borrowed_debug_tensor_runtime_ready_value;
        callback_invocation_ready_value_seen = callback_invocation_ready_value;
        full_package_consumer_report = full_package_consumer_report_value;
        return can_attempt_attach();
    }

    bool can_attempt_attach() const noexcept
    {
        return opt_in_enabled &&
            line_supports_debug_listener &&
            attach_guard_ready_value_seen &&
            native_vtable_ready_value_seen &&
            borrowed_debug_tensor_runtime_ready_value_seen &&
            callback_invocation_ready_value_seen;
    }

    bool attach_attempted() const noexcept
    {
        return can_attempt_attach();
    }

    bool attach_succeeded() const noexcept
    {
        return false;
    }

    bool detach_attempted() const noexcept
    {
        return attach_attempted();
    }

    bool detach_succeeded() const noexcept
    {
        return false;
    }

    bool rollback_attempted() const noexcept
    {
        return opt_in_enabled && !attach_succeeded();
    }

    bool rollback_succeeded() const noexcept
    {
        return rollback_attempted();
    }

    bool native_vtable_installed() const noexcept
    {
        return false;
    }

    bool process_debug_tensor_invoked() const noexcept
    {
        return false;
    }

    int32_t invocation_count() const noexcept
    {
        return 0;
    }

    int32_t failure_count() const noexcept
    {
        return opt_in_enabled && !attach_succeeded() ? 1 : 0;
    }

    int32_t in_flight_callback_count() const noexcept
    {
        return 0;
    }

    bool report_pointer_free() const noexcept
    {
        return true;
    }

    bool can_promote_real_callback_runtime() const noexcept
    {
        return full_package_consumer_report &&
            attach_succeeded() &&
            native_vtable_installed() &&
            process_debug_tensor_invoked() &&
            invocation_count() > 0 &&
            failure_count() == 0 &&
            in_flight_callback_count() == 0;
    }

    bool line_supports_debug_listener{};
    bool opt_in_enabled{};
    bool attach_guard_ready_value_seen{};
    bool native_vtable_ready_value_seen{};
    bool borrowed_debug_tensor_runtime_ready_value_seen{};
    bool callback_invocation_ready_value_seen{};
    bool full_package_consumer_report{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerRealNonNullAttachRuntimeSmokeAttempt>::value,
    "debug-listener-real-non-null-attach-runtime-smoke copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerRealNonNullAttachRuntimeSmokeAttempt>::value,
    "debug-listener-real-non-null-attach-runtime-smoke copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerRealNonNullAttachRuntimeSmokeAttempt>::value,
    "debug-listener-real-non-null-attach-runtime-smoke move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerRealNonNullAttachRuntimeSmokeAttempt>::value,
    "debug-listener-real-non-null-attach-runtime-smoke move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerRealNonNullAttachRuntimeSmokeAttempt>::value,
    "debug-listener-real-non-null-attach-runtime-smoke destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().configure_api_line(11)),
    "debug-listener-real-non-null-attach-runtime-smoke configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().configure_opt_in(false)),
    "debug-listener-real-non-null-attach-runtime-smoke configure_opt_in must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().configure_prerequisites(false, false, false, false, false)),
    "debug-listener-real-non-null-attach-runtime-smoke configure_prerequisites must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().can_attempt_attach()),
    "debug-listener-real-non-null-attach-runtime-smoke can_attempt_attach must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().attach_attempted()),
    "debug-listener-real-non-null-attach-runtime-smoke attach_attempted must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().attach_succeeded()),
    "debug-listener-real-non-null-attach-runtime-smoke attach_succeeded must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().detach_attempted()),
    "debug-listener-real-non-null-attach-runtime-smoke detach_attempted must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().detach_succeeded()),
    "debug-listener-real-non-null-attach-runtime-smoke detach_succeeded must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().rollback_attempted()),
    "debug-listener-real-non-null-attach-runtime-smoke rollback_attempted must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().rollback_succeeded()),
    "debug-listener-real-non-null-attach-runtime-smoke rollback_succeeded must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().native_vtable_installed()),
    "debug-listener-real-non-null-attach-runtime-smoke native_vtable_installed must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().process_debug_tensor_invoked()),
    "debug-listener-real-non-null-attach-runtime-smoke process_debug_tensor_invoked must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().invocation_count()),
    "debug-listener-real-non-null-attach-runtime-smoke invocation_count must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().failure_count()),
    "debug-listener-real-non-null-attach-runtime-smoke failure_count must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().in_flight_callback_count()),
    "debug-listener-real-non-null-attach-runtime-smoke in_flight_callback_count must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().report_pointer_free()),
    "debug-listener-real-non-null-attach-runtime-smoke report_pointer_free must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerRealNonNullAttachRuntimeSmokeAttempt&>().can_promote_real_callback_runtime()),
    "debug-listener-real-non-null-attach-runtime-smoke can_promote_real_callback_runtime must stay noexcept.");
} // namespace
