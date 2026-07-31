#ifndef JYPPX_TRT_DEBUG_LISTENER_OWNER_API
#error "JYPPX_TRT_DEBUG_LISTENER_OWNER_API must be defined before including debug_listener_callback_owner.inc"
#endif

#include <array>
#include <atomic>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace
{
constexpr const char* kDebugListenerOwnerFeature = "debug listener callback owner";
static std::atomic<uint64_t> g_debug_listener_owner_next_id{1U};

static void copy_debug_listener_text(char* destination, const size_t destination_size, const char* source) noexcept
{
    if (destination == nullptr || destination_size == 0U)
    {
        return;
    }

    const char* safe_source = source != nullptr ? source : "";
    const size_t source_length = std::strlen(safe_source);
    const size_t copy_length = source_length < destination_size - 1U ? source_length : destination_size - 1U;
    if (copy_length > 0U)
    {
        std::memcpy(destination, safe_source, copy_length);
    }

    destination[copy_length] = '\0';
}

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
static JYPPX_StatusCode set_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context,
    nvinfer1::IDebugListener* listener,
    JYPPX_Boolean* out_succeeded,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        *out_succeeded = context->setDebugListener(listener) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_succeeded = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_succeeded = context->setDebugListener(listener) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context,
    nvinfer1::IDebugListener** out_listener,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        *out_listener = context->getDebugListener();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_listener = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_listener = context->getDebugListener();
    return JYPPX_STATUS_OK;
#endif
}

class DebugListenerCallbackOwner final : public nvinfer1::IDebugListener
{
public:
    DebugListenerCallbackOwner(
        const JYPPX_TensorRtLine line,
        JYPPX_TensorRtDebugListenerCallback callback,
        void* user_state) noexcept
        : line_(line),
          owner_id_(g_debug_listener_owner_next_id.fetch_add(1U)),
          callback_(callback),
          user_state_(user_state)
    {
        copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "Native debug listener callback owner created in detached state.");
    }

    DebugListenerCallbackOwner(const DebugListenerCallbackOwner&) = delete;
    DebugListenerCallbackOwner& operator=(const DebugListenerCallbackOwner&) = delete;
    DebugListenerCallbackOwner(DebugListenerCallbackOwner&&) = delete;
    DebugListenerCallbackOwner& operator=(DebugListenerCallbackOwner&&) = delete;

    ~DebugListenerCallbackOwner() noexcept override
    {
        detach_noexcept();
    }

    bool processDebugTensor(
        void const* addr,
        nvinfer1::TensorLocation location,
        nvinfer1::DataType type,
        nvinfer1::Dims const& shape,
        char const* name,
        cudaStream_t stream) noexcept override
    {
        (void)addr;
        (void)stream;

        enter_callback();
        struct CallbackExit final
        {
            explicit CallbackExit(DebugListenerCallbackOwner* owner) noexcept : owner_(owner) {}
            ~CallbackExit() noexcept { owner_->leave_callback(); }
            DebugListenerCallbackOwner* owner_;
        } callback_exit(this);

        try
        {
            std::array<char, 256> copied_name{};
            copy_debug_listener_text(copied_name.data(), copied_name.size(), name);

            const int32_t shape_rank = shape.nbDims >= 0 && shape.nbDims <= 8 ? shape.nbDims : 0;
            std::array<int64_t, 8> copied_shape{};
            for (int32_t index = 0; index < shape_rank; ++index)
            {
                copied_shape[static_cast<size_t>(index)] = static_cast<int64_t>(shape.d[index]);
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                copy_debug_listener_text(last_tensor_name_, sizeof(last_tensor_name_), copied_name.data());
                last_data_type_ = static_cast<int32_t>(type);
                last_location_ = static_cast<int32_t>(location);
                last_shape_rank_ = shape_rank;
                last_shape_ = copied_shape;
            }

            invocation_count_.fetch_add(1U);
            const auto callback_status = callback_ != nullptr
                ? callback_(
                    static_cast<uint32_t>(line_),
                    copied_name.data(),
                    std::strlen(copied_name.data()),
                    static_cast<int32_t>(type),
                    static_cast<int32_t>(location),
                    shape_rank,
                    copied_shape[0], copied_shape[1], copied_shape[2], copied_shape[3],
                    copied_shape[4], copied_shape[5], copied_shape[6], copied_shape[7],
                    user_state_)
                : JYPPX_STATUS_INVALID_STATE;

            const bool succeeded = callback_status == JYPPX_STATUS_OK;
            if (!succeeded)
            {
                failure_count_.fetch_add(1U);
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                last_status_ = callback_status;
                last_callback_succeeded_ = succeeded;
                copy_debug_listener_text(
                    last_diagnostic_,
                    sizeof(last_diagnostic_),
                    succeeded
                        ? "TensorRT invoked IDebugListener::processDebugTensor; copied metadata reached the managed callback."
                        : "TensorRT invoked IDebugListener::processDebugTensor; the managed callback returned a failure status.");
            }

            return succeeded;
        }
        catch (const std::exception& exception)
        {
            record_callback_exception(exception.what());
            return false;
        }
        catch (...)
        {
            record_callback_exception("unknown native exception");
            return false;
        }
    }

    JYPPX_StatusCode attach(nvinfer1::IExecutionContext* context, JYPPX_Boolean* out_attached)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (attached_context_ != nullptr && attached_context_ != context)
        {
            copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "Debug listener owner is already attached to another execution context.");
            last_status_ = JYPPX_STATUS_INVALID_STATE;
            return JYPPX_STATUS_INVALID_STATE;
        }

        if (attached_context_ == context)
        {
            *out_attached = JYPPX_TRUE;
            return JYPPX_STATUS_OK;
        }

        JYPPX_Boolean attached = JYPPX_FALSE;
        const auto status = set_debug_listener_with_seh_guard(context, this, &attached, "execution context debug listener owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            last_status_ = status;
            copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "TensorRT debug listener attach raised a guarded native failure.");
            return status;
        }

        if (attached == JYPPX_FALSE)
        {
            last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
            copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "IExecutionContext::setDebugListener returned false.");
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
            return JYPPX_STATUS_RUNTIME_ERROR;
        }

        attached_context_ = context;
        attach_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_OK;
        copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "Native debug listener vtable attached to the TensorRT execution context.");
        *out_attached = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
    }

    JYPPX_StatusCode detach(JYPPX_Boolean* out_detached)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (attached_context_ == nullptr)
        {
            *out_detached = JYPPX_TRUE;
            return JYPPX_STATUS_OK;
        }

        nvinfer1::IDebugListener* active_listener = nullptr;
        auto status = get_debug_listener_with_seh_guard(
            attached_context_,
            &active_listener,
            "execution context debug listener owner detach query");
        if (status != JYPPX_STATUS_OK)
        {
            last_status_ = status;
            return status;
        }

        if (active_listener == this)
        {
            JYPPX_Boolean cleared = JYPPX_FALSE;
            status = set_debug_listener_with_seh_guard(
                attached_context_,
                nullptr,
                &cleared,
                "execution context debug listener owner detach");
            if (status != JYPPX_STATUS_OK)
            {
                last_status_ = status;
                return status;
            }

            if (cleared == JYPPX_FALSE)
            {
                last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
                copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "IExecutionContext::setDebugListener(nullptr) returned false.");
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
                return JYPPX_STATUS_RUNTIME_ERROR;
            }
        }

        attached_context_ = nullptr;
        detach_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_OK;
        copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), "Native debug listener detached before callback owner release.");
        *out_detached = JYPPX_TRUE;
        callback_drained_.wait(lock, [this]() noexcept { return in_flight_callback_count_.load() == 0U; });
        return JYPPX_STATUS_OK;
    }

    void copy_info(JYPPX_TensorRtDebugListenerOwnerInfo* out_info) const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        out_info->line = static_cast<uint32_t>(line_);
        out_info->owner_id = owner_id_;
        out_info->invocation_count = invocation_count_.load();
        out_info->failure_count = failure_count_.load();
        out_info->in_flight_callback_count = in_flight_callback_count_.load();
        out_info->max_in_flight_callback_count = max_in_flight_callback_count_.load();
        out_info->attach_count = attach_count_.load();
        out_info->detach_count = detach_count_.load();
        out_info->last_status = static_cast<int32_t>(last_status_);
        out_info->is_attached = attached_context_ != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_callback_succeeded = last_callback_succeeded_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_data_type = last_data_type_;
        out_info->last_location = last_location_;
        out_info->last_shape_rank = last_shape_rank_;
        for (size_t index = 0U; index < last_shape_.size(); ++index)
        {
            out_info->last_shape[index] = last_shape_[index];
        }

        copy_debug_listener_text(out_info->last_tensor_name, sizeof(out_info->last_tensor_name), last_tensor_name_);
        copy_debug_listener_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic_);
    }

private:
    void enter_callback() noexcept
    {
        const uint64_t current = in_flight_callback_count_.fetch_add(1U) + 1U;
        uint64_t observed = max_in_flight_callback_count_.load();
        while (current > observed && !max_in_flight_callback_count_.compare_exchange_weak(observed, current))
        {
        }
    }

    void leave_callback() noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            in_flight_callback_count_.fetch_sub(1U);
        }
        catch (...)
        {
            in_flight_callback_count_.fetch_sub(1U);
        }

        callback_drained_.notify_all();
    }

    void record_callback_exception(const char* detail) noexcept
    {
        failure_count_.fetch_add(1U);
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
            last_callback_succeeded_ = false;
            copy_debug_listener_text(last_diagnostic_, sizeof(last_diagnostic_), detail);
        }
        catch (...)
        {
        }
    }

    void detach_noexcept() noexcept
    {
        try
        {
            JYPPX_Boolean detached = JYPPX_FALSE;
            (void)detach(&detached);
        }
        catch (...)
        {
        }
    }

    JYPPX_TensorRtLine line_;
    uint64_t owner_id_;
    JYPPX_TensorRtDebugListenerCallback callback_;
    void* user_state_;
    mutable std::mutex state_mutex_;
    std::condition_variable callback_drained_;
    nvinfer1::IExecutionContext* attached_context_{nullptr};
    std::atomic<uint64_t> invocation_count_{0U};
    std::atomic<uint64_t> failure_count_{0U};
    std::atomic<uint64_t> in_flight_callback_count_{0U};
    std::atomic<uint64_t> max_in_flight_callback_count_{0U};
    std::atomic<uint64_t> attach_count_{0U};
    std::atomic<uint64_t> detach_count_{0U};
    JYPPX_StatusCode last_status_{JYPPX_STATUS_OK};
    bool last_callback_succeeded_{false};
    int32_t last_data_type_{-1};
    int32_t last_location_{-1};
    int32_t last_shape_rank_{0};
    std::array<int64_t, 8> last_shape_{};
    char last_tensor_name_[256]{};
    char last_diagnostic_[1024]{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerCallbackOwner>::value &&
        !std::is_copy_assignable<DebugListenerCallbackOwner>::value,
    "Debug listener callback owners must not be copyable.");
static_assert(
    !std::is_move_constructible<DebugListenerCallbackOwner>::value &&
        !std::is_move_assignable<DebugListenerCallbackOwner>::value,
    "Debug listener callback owners must not be movable.");
static_assert(
    std::is_nothrow_destructible<DebugListenerCallbackOwner>::value,
    "Debug listener callback owner destruction must not throw.");
static_assert(
    noexcept(std::declval<DebugListenerCallbackOwner&>().processDebugTensor(
        std::declval<void const*>(),
        std::declval<nvinfer1::TensorLocation>(),
        std::declval<nvinfer1::DataType>(),
        std::declval<nvinfer1::Dims const&>(),
        std::declval<char const*>(),
        std::declval<cudaStream_t>())),
    "IDebugListener::processDebugTensor must not throw across the native vtable.");

static void destroy_debug_listener_callback_owner(void* payload) noexcept
{
    delete static_cast<DebugListenerCallbackOwner*>(payload);
}

static JYPPX_StatusCode get_debug_listener_callback_owner(
    JYPPX_TensorRtDebugListenerOwner* owner,
    DebugListenerCallbackOwner** out_payload)
{
    auto status = jyppx::tensorrt::validate_handle(
        owner,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_DEBUG_LISTENER_CALLBACK_OWNER,
        "owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<DebugListenerCallbackOwner>(owner);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Debug listener owner handle has no callback payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_DEBUG_LISTENER_OWNER_API(debug_listener_owner_create)(
    JYPPX_TensorRtDebugListenerCallback callback,
    void* user_state,
    JYPPX_TensorRtDebugListenerOwner** out_owner)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_owner, "out_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_owner = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Debug listener callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    try
    {
        auto* payload = new (std::nothrow) DebugListenerCallbackOwner(kLine, callback, user_state);
        if (payload == nullptr)
        {
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_DEBUG_LISTENER_CALLBACK_OWNER,
            payload,
            &destroy_debug_listener_callback_owner);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        *out_owner = reinterpret_cast<JYPPX_TensorRtDebugListenerOwner*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)user_state;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kDebugListenerOwnerFeature);
#else
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, kDebugListenerOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_DEBUG_LISTENER_OWNER_API(debug_listener_owner_attach)(
    JYPPX_TensorRtDebugListenerOwner* owner,
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_attached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attached, "out_attached");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_attached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    try
    {
        DebugListenerCallbackOwner* owner_payload = nullptr;
        status = get_debug_listener_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        nvinfer1::IExecutionContext* context_payload = nullptr;
        status = get_context_payload_ext(context, &context_payload, "execution context debug listener owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        return owner_payload->attach(context_payload, out_attached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kDebugListenerOwnerFeature);
#else
    (void)owner;
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, kDebugListenerOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_DEBUG_LISTENER_OWNER_API(debug_listener_owner_detach)(
    JYPPX_TensorRtDebugListenerOwner* owner,
    JYPPX_Boolean* out_detached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_detached, "out_detached");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_detached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    try
    {
        DebugListenerCallbackOwner* owner_payload = nullptr;
        status = get_debug_listener_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        return owner_payload->detach(out_detached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kDebugListenerOwnerFeature);
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kDebugListenerOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_DEBUG_LISTENER_OWNER_API(debug_listener_owner_get_info)(
    JYPPX_TensorRtDebugListenerOwner* owner,
    JYPPX_TensorRtDebugListenerOwnerInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_info, 0, sizeof(*out_info));
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    try
    {
        DebugListenerCallbackOwner* owner_payload = nullptr;
        status = get_debug_listener_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        owner_payload->copy_info(out_info);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kDebugListenerOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kDebugListenerOwnerFeature);
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kDebugListenerOwnerFeature);
#endif
}
