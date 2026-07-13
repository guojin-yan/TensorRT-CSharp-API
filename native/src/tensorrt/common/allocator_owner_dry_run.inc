#ifndef JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API
#error "JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API must be defined before including allocator_owner_dry_run.inc"
#endif

#include <atomic>

namespace
{
constexpr const char* kAllocatorOwnerDryRunFeature = "allocator owner native dry-run controls";
constexpr int32_t kAllocatorOwnerAttachStateDetached = 0;
constexpr int32_t kAllocatorOwnerAttachStateIntentRecorded = 1;

static std::atomic<uint64_t> g_allocator_owner_dry_run_next_id{1U};

static void copy_allocator_owner_diagnostic(char* destination, const size_t destination_size, const char* value) noexcept
{
    if (destination == nullptr || destination_size == 0U)
    {
        return;
    }

    const char* safe_value = value != nullptr ? value : "";
    const size_t length = std::strlen(safe_value);
    const size_t copy_length = length < (destination_size - 1U)
        ? length
        : (destination_size - 1U);
    if (copy_length != 0U)
    {
        std::memcpy(destination, safe_value, copy_length);
    }

    destination[copy_length] = '\0';
}

struct AllocatorOwnerDryRunPayload final
{
    AllocatorOwnerDryRunPayload(const AllocatorOwnerDryRunPayload&) = delete;
    AllocatorOwnerDryRunPayload& operator=(const AllocatorOwnerDryRunPayload&) = delete;

    explicit AllocatorOwnerDryRunPayload(const JYPPX_TensorRtLine owner_line) noexcept
        : line(owner_line),
          owner_id(g_allocator_owner_dry_run_next_id.fetch_add(1U))
    {
        copy_operation("create");
        copy_diagnostic("native dry-run owner created; not attached to TensorRT allocator callback.");
    }

    ~AllocatorOwnerDryRunPayload() noexcept = default;

    void copy_diagnostic(const char* value) noexcept
    {
        copy_allocator_owner_diagnostic(last_diagnostic, sizeof(last_diagnostic), value);
    }

    void copy_operation(const char* value) noexcept
    {
        copy_allocator_owner_diagnostic(last_operation, sizeof(last_operation), value);
    }

    JYPPX_StatusCode emit_diagnostic(const uint64_t size, const uint64_t alignment, const char* reason)
    {
        ++invocation_count;
        last_size = size;
        last_alignment = alignment;
        copy_operation("emit-diagnostic");

        if (alignment == 0U)
        {
            ++failure_count;
            last_status = JYPPX_STATUS_INVALID_ARGUMENT;
            copy_diagnostic("allocator-owner-native-dry-run-controls invalid alignment; no TensorRT allocator callback was invoked.");
            return last_status;
        }

        const char* safe_reason = (reason != nullptr && reason[0] != '\0') ? reason : "unspecified";
        std::ostringstream builder;
        builder << "allocator-owner-native-dry-run-controls line=" << static_cast<uint32_t>(line)
                << " size=" << size
                << " alignment=" << alignment
                << " reason=" << safe_reason
                << "; no TensorRT allocator callback was invoked.";
        last_status = JYPPX_STATUS_OK;
        copy_diagnostic(builder.str().c_str());
        return last_status;
    }

    JYPPX_StatusCode attach_intent(const char* target_kind)
    {
        ++state_transition_count;
        attach_state = kAllocatorOwnerAttachStateIntentRecorded;
        last_status = JYPPX_STATUS_OK;
        copy_operation("attach-intent");

        const char* safe_target = (target_kind != nullptr && target_kind[0] != '\0')
            ? target_kind
            : "unspecified-target";
        std::ostringstream builder;
        builder << "allocator-owner-state-ledger-dry-run-controls owner=" << owner_id
                << " attach-intent target=" << safe_target
                << "; TensorRT setGpuAllocator/setOutputAllocator was not called.";
        copy_diagnostic(builder.str().c_str());
        return last_status;
    }

    JYPPX_StatusCode detach_intent(const char* target_kind)
    {
        ++state_transition_count;
        attach_state = kAllocatorOwnerAttachStateDetached;
        last_status = JYPPX_STATUS_OK;
        copy_operation("detach-intent");

        const char* safe_target = (target_kind != nullptr && target_kind[0] != '\0')
            ? target_kind
            : "unspecified-target";
        std::ostringstream builder;
        builder << "allocator-owner-state-ledger-dry-run-controls owner=" << owner_id
                << " detach-intent target=" << safe_target
                << "; native owner remains detached from TensorRT.";
        copy_diagnostic(builder.str().c_str());
        return last_status;
    }

    JYPPX_StatusCode record_allocation_intent(
        const uint64_t size,
        const uint64_t alignment,
        const uint64_t stream_value)
    {
        last_size = size;
        last_alignment = alignment;
        last_stream_value = stream_value;
        copy_operation("allocation-intent");

        if (alignment == 0U)
        {
            ++ledger_failure_count;
            last_status = JYPPX_STATUS_INVALID_ARGUMENT;
            copy_diagnostic("allocator-owner-state-ledger-dry-run-controls invalid allocation alignment; no device pointer was requested.");
            return last_status;
        }

        ++ledger_allocation_count;
        last_allocation_id = next_allocation_id++;
        has_live_allocation = true;
        last_status = JYPPX_STATUS_OK;

        std::ostringstream builder;
        builder << "allocator-owner-state-ledger-dry-run-controls owner=" << owner_id
                << " allocation-intent id=" << last_allocation_id
                << " size=" << size
                << " alignment=" << alignment
                << " stream=" << stream_value
                << "; no device pointer was allocated.";
        copy_diagnostic(builder.str().c_str());
        return last_status;
    }

    JYPPX_StatusCode record_release_intent(const uint64_t allocation_id, const uint64_t stream_value)
    {
        last_release_allocation_id = allocation_id;
        last_stream_value = stream_value;
        copy_operation("release-intent");

        if (allocation_id == 0U)
        {
            ++ledger_failure_count;
            last_status = JYPPX_STATUS_INVALID_ARGUMENT;
            copy_diagnostic("allocator-owner-state-ledger-dry-run-controls invalid release allocation id; no device pointer was released.");
            return last_status;
        }

        if (!has_live_allocation || allocation_id != last_allocation_id)
        {
            ++ledger_failure_count;
            last_status = JYPPX_STATUS_INVALID_STATE;
            copy_diagnostic("allocator-owner-state-ledger-dry-run-controls release intent did not match the live synthetic allocation.");
            return last_status;
        }

        ++ledger_release_count;
        has_live_allocation = false;
        last_status = JYPPX_STATUS_OK;

        std::ostringstream builder;
        builder << "allocator-owner-state-ledger-dry-run-controls owner=" << owner_id
                << " release-intent id=" << allocation_id
                << " stream=" << stream_value
                << "; no device pointer was released.";
        copy_diagnostic(builder.str().c_str());
        return last_status;
    }

    void copy_to(JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info) const noexcept
    {
        out_info->line = static_cast<uint32_t>(line);
        out_info->invocation_count = invocation_count;
        out_info->failure_count = failure_count;
        out_info->last_status = static_cast<int32_t>(last_status);
        out_info->is_attached = JYPPX_FALSE;
        out_info->last_size = last_size;
        out_info->last_alignment = last_alignment;
        copy_allocator_owner_diagnostic(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic);
    }

    void copy_state_to(JYPPX_TensorRtAllocatorOwnerStateInfo* out_info) const noexcept
    {
        out_info->line = static_cast<uint32_t>(line);
        out_info->owner_id = owner_id;
        out_info->state_transition_count = state_transition_count;
        out_info->ledger_allocation_count = ledger_allocation_count;
        out_info->ledger_release_count = ledger_release_count;
        out_info->ledger_failure_count = ledger_failure_count;
        out_info->last_allocation_id = last_allocation_id;
        out_info->last_release_allocation_id = last_release_allocation_id;
        out_info->last_size = last_size;
        out_info->last_alignment = last_alignment;
        out_info->last_stream_value = last_stream_value;
        out_info->attach_state = attach_state;
        out_info->last_status = static_cast<int32_t>(last_status);
        out_info->is_attached = JYPPX_FALSE;
        out_info->has_live_allocation = has_live_allocation ? JYPPX_TRUE : JYPPX_FALSE;
        copy_allocator_owner_diagnostic(out_info->last_operation, sizeof(out_info->last_operation), last_operation);
        copy_allocator_owner_diagnostic(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic);
    }

    JYPPX_TensorRtLine line;
    uint64_t owner_id{0U};
    uint64_t invocation_count{0U};
    uint64_t failure_count{0U};
    uint64_t state_transition_count{0U};
    uint64_t ledger_allocation_count{0U};
    uint64_t ledger_release_count{0U};
    uint64_t ledger_failure_count{0U};
    uint64_t next_allocation_id{1U};
    uint64_t last_allocation_id{0U};
    uint64_t last_release_allocation_id{0U};
    JYPPX_StatusCode last_status{JYPPX_STATUS_OK};
    uint64_t last_size{0U};
    uint64_t last_alignment{0U};
    uint64_t last_stream_value{0U};
    int32_t attach_state{kAllocatorOwnerAttachStateDetached};
    bool has_live_allocation{false};
    char last_operation[64]{};
    char last_diagnostic[1024]{};
};

static void reset_allocator_owner_diagnostic_info(JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    out_info->line = static_cast<uint32_t>(kLine);
    out_info->invocation_count = 0U;
    out_info->failure_count = 0U;
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_OK);
    out_info->is_attached = JYPPX_FALSE;
    out_info->last_size = 0U;
    out_info->last_alignment = 0U;
    out_info->last_diagnostic[0] = '\0';
}

static void reset_allocator_owner_state_info(JYPPX_TensorRtAllocatorOwnerStateInfo* out_info) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    out_info->line = static_cast<uint32_t>(kLine);
    out_info->owner_id = 0U;
    out_info->state_transition_count = 0U;
    out_info->ledger_allocation_count = 0U;
    out_info->ledger_release_count = 0U;
    out_info->ledger_failure_count = 0U;
    out_info->last_allocation_id = 0U;
    out_info->last_release_allocation_id = 0U;
    out_info->last_size = 0U;
    out_info->last_alignment = 0U;
    out_info->last_stream_value = 0U;
    out_info->attach_state = kAllocatorOwnerAttachStateDetached;
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_OK);
    out_info->is_attached = JYPPX_FALSE;
    out_info->has_live_allocation = JYPPX_FALSE;
    out_info->last_operation[0] = '\0';
    out_info->last_diagnostic[0] = '\0';
}

static void destroy_allocator_owner_dry_run_payload(void* payload) noexcept
{
    delete static_cast<AllocatorOwnerDryRunPayload*>(payload);
}

static JYPPX_StatusCode get_allocator_owner_dry_run_payload(
    JYPPX_TensorRtAllocatorOwner* owner,
    AllocatorOwnerDryRunPayload** out_payload)
{
    auto status = jyppx::tensorrt::validate_handle(
        owner,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_ALLOCATOR_CALLBACK_OWNER,
        "allocator_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<AllocatorOwnerDryRunPayload>(owner);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Allocator owner dry-run handle does not carry a diagnostic payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_create)(
    JYPPX_TensorRtAllocatorOwner** out_owner)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_owner, "out_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_owner = nullptr;

    try
    {
        auto* payload = new (std::nothrow) AllocatorOwnerDryRunPayload(kLine);
        if (payload == nullptr)
        {
            jyppx::common::set_last_error(
                JYPPX_ERROR_CATEGORY_TENSORRT,
                "Unable to allocate native allocator owner dry-run payload.");
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_ALLOCATOR_CALLBACK_OWNER,
            payload,
            &destroy_allocator_owner_dry_run_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        *out_owner = reinterpret_cast<JYPPX_TensorRtAllocatorOwner*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_emit_diagnostic)(
    JYPPX_TensorRtAllocatorOwner* owner,
    uint64_t size,
    uint64_t alignment,
    const char* reason,
    JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_diagnostic_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = payload->emit_diagnostic(size, alignment, reason);
        payload->copy_to(out_info);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, out_info->last_diagnostic);
        }

        return status;
    }
    catch (const std::exception& exception)
    {
        reset_allocator_owner_diagnostic_info(out_info);
        out_info->failure_count = 1U;
        out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_RUNTIME_ERROR);
        std::ostringstream builder;
        builder << "allocator-owner-native-dry-run-controls caught native exception: " << exception.what();
        const std::string message = builder.str();
        copy_allocator_owner_diagnostic(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), message.c_str());
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        reset_allocator_owner_diagnostic_info(out_info);
        out_info->failure_count = 1U;
        out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_RUNTIME_ERROR);
        copy_allocator_owner_diagnostic(
            out_info->last_diagnostic,
            sizeof(out_info->last_diagnostic),
            "allocator-owner-native-dry-run-controls caught unknown native exception."
        );
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_get_info)(
    JYPPX_TensorRtAllocatorOwner* owner,
    JYPPX_TensorRtAllocatorOwnerDiagnosticInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_diagnostic_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        payload->copy_to(out_info);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_get_state)(
    JYPPX_TensorRtAllocatorOwner* owner,
    JYPPX_TensorRtAllocatorOwnerStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_state_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        payload->copy_state_to(out_info);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_attach_intent)(
    JYPPX_TensorRtAllocatorOwner* owner,
    const char* target_kind,
    JYPPX_TensorRtAllocatorOwnerStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_state_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = payload->attach_intent(target_kind);
        payload->copy_state_to(out_info);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, out_info->last_diagnostic);
        }

        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_detach_intent)(
    JYPPX_TensorRtAllocatorOwner* owner,
    const char* target_kind,
    JYPPX_TensorRtAllocatorOwnerStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_state_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = payload->detach_intent(target_kind);
        payload->copy_state_to(out_info);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, out_info->last_diagnostic);
        }

        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_ledger_record_allocation_intent)(
    JYPPX_TensorRtAllocatorOwner* owner,
    uint64_t size,
    uint64_t alignment,
    uint64_t stream_value,
    JYPPX_TensorRtAllocatorOwnerStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_state_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = payload->record_allocation_intent(size, alignment, stream_value);
        payload->copy_state_to(out_info);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, out_info->last_diagnostic);
        }

        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}

JYPPX_StatusCode JYPPX_TRT_ALLOCATOR_OWNER_DRY_RUN_API(allocator_owner_dry_run_ledger_record_release_intent)(
    JYPPX_TensorRtAllocatorOwner* owner,
    uint64_t allocation_id,
    uint64_t stream_value,
    JYPPX_TensorRtAllocatorOwnerStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_allocator_owner_state_info(out_info);

    try
    {
        AllocatorOwnerDryRunPayload* payload = nullptr;
        status = get_allocator_owner_dry_run_payload(owner, &payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = payload->record_release_intent(allocation_id, stream_value);
        payload->copy_state_to(out_info);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, out_info->last_diagnostic);
        }

        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kAllocatorOwnerDryRunFeature, "unknown native exception");
    }
}
