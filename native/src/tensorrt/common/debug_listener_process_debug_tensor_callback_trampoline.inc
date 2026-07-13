#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerBorrowedDebugTensorMetadataCopy final
{
    DebugListenerBorrowedDebugTensorMetadataCopy() noexcept = default;
    DebugListenerBorrowedDebugTensorMetadataCopy(const DebugListenerBorrowedDebugTensorMetadataCopy&) = delete;
    DebugListenerBorrowedDebugTensorMetadataCopy& operator=(const DebugListenerBorrowedDebugTensorMetadataCopy&) = delete;
    DebugListenerBorrowedDebugTensorMetadataCopy(DebugListenerBorrowedDebugTensorMetadataCopy&&) = delete;
    DebugListenerBorrowedDebugTensorMetadataCopy& operator=(DebugListenerBorrowedDebugTensorMetadataCopy&&) = delete;
    ~DebugListenerBorrowedDebugTensorMetadataCopy() noexcept = default;

    bool configure_metadata(
        bool name_copied_value,
        bool type_copied_value,
        bool location_copied_value,
        bool shape_copied_value,
        bool flags_copied_value) noexcept
    {
        name_copied = name_copied_value;
        type_copied = type_copied_value;
        location_copied = location_copied_value;
        shape_copied = shape_copied_value;
        flags_copied = flags_copied_value;
        return metadata_copy_ready();
    }

    bool metadata_copy_ready() const noexcept
    {
        return name_copied && type_copied && location_copied && shape_copied && flags_copied &&
            !borrowed_debug_tensor_pointer_exposed && !borrowed_debug_tensor_data_pointer_exposed;
    }

    bool name_copied{};
    bool type_copied{};
    bool location_copied{};
    bool shape_copied{};
    bool flags_copied{};
    bool borrowed_debug_tensor_pointer_exposed{};
    bool borrowed_debug_tensor_data_pointer_exposed{};
};

struct DebugListenerCallbackInFlightScope final
{
    DebugListenerCallbackInFlightScope() noexcept = default;
    DebugListenerCallbackInFlightScope(const DebugListenerCallbackInFlightScope&) = delete;
    DebugListenerCallbackInFlightScope& operator=(const DebugListenerCallbackInFlightScope&) = delete;
    DebugListenerCallbackInFlightScope(DebugListenerCallbackInFlightScope&&) = delete;
    DebugListenerCallbackInFlightScope& operator=(DebugListenerCallbackInFlightScope&&) = delete;
    ~DebugListenerCallbackInFlightScope() noexcept = default;

    bool enter() noexcept
    {
        ++entry_count;
        ++in_flight_count;
        if (in_flight_count > max_in_flight_count)
        {
            max_in_flight_count = in_flight_count;
        }

        return true;
    }

    bool leave_success() noexcept
    {
        if (in_flight_count > 0)
        {
            --in_flight_count;
        }

        ++leave_count;
        return in_flight_count >= 0;
    }

    bool leave_failure(int32_t status) noexcept
    {
        last_status = status;
        ++failure_count;
        return leave_success();
    }

    bool balanced() const noexcept
    {
        return entry_count == leave_count && in_flight_count == 0 && max_in_flight_count >= 0;
    }

    int32_t entry_count{};
    int32_t leave_count{};
    int32_t failure_count{};
    int32_t in_flight_count{};
    int32_t max_in_flight_count{};
    int32_t last_status{};
};

struct DebugListenerProcessDebugTensorCallbackReport final
{
    bool trampoline_shape_ready{};
    bool native_callback_entry_located{};
    bool no_throw_callback_entry_ready{};
    bool exception_capture_ready{};
    bool callback_status_mapping_ready{};
    bool in_flight_accounting_ready{};
    bool detach_before_release_ready{};
    bool borrowed_debug_tensor_metadata_copy_ready{};
    bool borrowed_debug_tensor_pointer_exposed{};
    bool borrowed_debug_tensor_data_pointer_exposed{};
    bool pointer_free_surface_ready{};
    bool process_debug_tensor_runtime_ready{};
    int32_t invocation_count{};
    int32_t failure_count{};
    int32_t in_flight_callback_count{};
    int32_t last_status{};
};

struct DebugListenerProcessDebugTensorCallbackTrampoline final
{
    DebugListenerProcessDebugTensorCallbackTrampoline() noexcept = default;
    DebugListenerProcessDebugTensorCallbackTrampoline(const DebugListenerProcessDebugTensorCallbackTrampoline&) = delete;
    DebugListenerProcessDebugTensorCallbackTrampoline& operator=(const DebugListenerProcessDebugTensorCallbackTrampoline&) = delete;
    DebugListenerProcessDebugTensorCallbackTrampoline(DebugListenerProcessDebugTensorCallbackTrampoline&&) = delete;
    DebugListenerProcessDebugTensorCallbackTrampoline& operator=(DebugListenerProcessDebugTensorCallbackTrampoline&&) = delete;
    ~DebugListenerProcessDebugTensorCallbackTrampoline() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool configure_shape(
        bool no_throw_entry,
        bool exception_capture,
        bool status_mapping,
        bool in_flight_accounting,
        bool metadata_copy,
        bool detach_before_release) noexcept
    {
        no_throw_callback_entry_ready = no_throw_entry;
        exception_capture_ready = exception_capture;
        callback_status_mapping_ready = status_mapping;
        in_flight_accounting_ready = in_flight_accounting;
        borrowed_debug_tensor_metadata_copy_ready = metadata_copy;
        detach_before_release_ready = detach_before_release;
        return trampoline_shape_ready();
    }

    bool begin_callback() noexcept
    {
        callback_scope.enter();
        native_callback_entry_located = line_supports_debug_listener;
        return can_return_status_without_throwing();
    }

    bool complete_callback_success() noexcept
    {
        return callback_scope.leave_success();
    }

    bool complete_callback_failure(int32_t status) noexcept
    {
        return callback_scope.leave_failure(status);
    }

    bool can_return_status_without_throwing() const noexcept
    {
        return no_throw_callback_entry_ready && exception_capture_ready && callback_status_mapping_ready;
    }

    bool trampoline_shape_ready() const noexcept
    {
        return line_supports_debug_listener &&
            native_callback_entry_located &&
            no_throw_callback_entry_ready &&
            exception_capture_ready &&
            callback_status_mapping_ready &&
            in_flight_accounting_ready &&
            detach_before_release_ready &&
            borrowed_debug_tensor_metadata_copy_ready &&
            !borrowed_debug_tensor_pointer_exposed &&
            !borrowed_debug_tensor_data_pointer_exposed;
    }

    DebugListenerProcessDebugTensorCallbackReport make_report() const noexcept
    {
        DebugListenerProcessDebugTensorCallbackReport report{};
        report.trampoline_shape_ready = trampoline_shape_ready();
        report.native_callback_entry_located = native_callback_entry_located;
        report.no_throw_callback_entry_ready = no_throw_callback_entry_ready;
        report.exception_capture_ready = exception_capture_ready;
        report.callback_status_mapping_ready = callback_status_mapping_ready;
        report.in_flight_accounting_ready = in_flight_accounting_ready && callback_scope.balanced();
        report.detach_before_release_ready = detach_before_release_ready;
        report.borrowed_debug_tensor_metadata_copy_ready = borrowed_debug_tensor_metadata_copy_ready;
        report.borrowed_debug_tensor_pointer_exposed = borrowed_debug_tensor_pointer_exposed;
        report.borrowed_debug_tensor_data_pointer_exposed = borrowed_debug_tensor_data_pointer_exposed;
        report.pointer_free_surface_ready = !borrowed_debug_tensor_pointer_exposed && !borrowed_debug_tensor_data_pointer_exposed;
        report.process_debug_tensor_runtime_ready = false;
        report.invocation_count = 0;
        report.failure_count = callback_scope.failure_count;
        report.in_flight_callback_count = callback_scope.in_flight_count;
        report.last_status = callback_scope.last_status;
        return report;
    }

    bool line_supports_debug_listener{};
    bool native_callback_entry_located{};
    bool no_throw_callback_entry_ready{};
    bool exception_capture_ready{};
    bool callback_status_mapping_ready{};
    bool in_flight_accounting_ready{};
    bool detach_before_release_ready{};
    bool borrowed_debug_tensor_metadata_copy_ready{};
    bool borrowed_debug_tensor_pointer_exposed{};
    bool borrowed_debug_tensor_data_pointer_exposed{};
    DebugListenerCallbackInFlightScope callback_scope{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerProcessDebugTensorCallbackTrampoline>::value,
    "debug-listener-process-debug-tensor-callback-trampoline copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerProcessDebugTensorCallbackTrampoline>::value,
    "debug-listener-process-debug-tensor-callback-trampoline copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerProcessDebugTensorCallbackTrampoline>::value,
    "debug-listener-process-debug-tensor-callback-trampoline move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerProcessDebugTensorCallbackTrampoline>::value,
    "debug-listener-process-debug-tensor-callback-trampoline move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerProcessDebugTensorCallbackTrampoline>::value,
    "debug-listener-process-debug-tensor-callback-trampoline destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerProcessDebugTensorCallbackTrampoline&>().configure_api_line(11)),
    "debug-listener-process-debug-tensor-callback-trampoline configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerProcessDebugTensorCallbackTrampoline&>().configure_shape(false, false, false, false, false, false)),
    "debug-listener-process-debug-tensor-callback-trampoline configure_shape must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerProcessDebugTensorCallbackTrampoline&>().begin_callback()),
    "debug-listener-process-debug-tensor-callback-trampoline begin_callback must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerProcessDebugTensorCallbackTrampoline&>().complete_callback_success()),
    "debug-listener-process-debug-tensor-callback-trampoline complete_callback_success must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerProcessDebugTensorCallbackTrampoline&>().complete_callback_failure(-1)),
    "debug-listener-process-debug-tensor-callback-trampoline complete_callback_failure must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerProcessDebugTensorCallbackTrampoline&>().can_return_status_without_throwing()),
    "debug-listener-process-debug-tensor-callback-trampoline can_return_status_without_throwing must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerProcessDebugTensorCallbackTrampoline&>().trampoline_shape_ready()),
    "debug-listener-process-debug-tensor-callback-trampoline trampoline_shape_ready must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerProcessDebugTensorCallbackTrampoline&>().make_report()),
    "debug-listener-process-debug-tensor-callback-trampoline make_report must stay noexcept.");
} // namespace
