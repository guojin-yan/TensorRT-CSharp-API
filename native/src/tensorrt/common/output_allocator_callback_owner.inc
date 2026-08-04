#ifndef JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API
#error "JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API must be defined before including output_allocator_callback_owner.inc"
#endif

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifndef JYPPX_HAS_CUDA_TOOLKIT
#define JYPPX_HAS_CUDA_TOOLKIT 0
#endif

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace
{
constexpr const char* kOutputAllocatorOwnerFeature = "output allocator callback owner";
static std::atomic<uint64_t> g_output_allocator_owner_next_id{1U};

static void copy_output_allocator_text(char* destination, const size_t destination_size, const char* source) noexcept
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

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
static JYPPX_StatusCode set_output_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context,
    const char* tensor_name,
    nvinfer1::IOutputAllocator* allocator,
    JYPPX_Boolean* out_succeeded,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        *out_succeeded = context->setOutputAllocator(tensor_name, allocator) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_succeeded = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_succeeded = context->setOutputAllocator(tensor_name, allocator) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_output_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context,
    const char* tensor_name,
    nvinfer1::IOutputAllocator** out_allocator,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        *out_allocator = context->getOutputAllocator(tensor_name);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_allocator = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_allocator = context->getOutputAllocator(tensor_name);
    return JYPPX_STATUS_OK;
#endif
}

class OutputAllocatorCallbackOwner final : public nvinfer1::IOutputAllocator
{
public:
    OutputAllocatorCallbackOwner(
        const JYPPX_TensorRtLine line,
        JYPPX_TensorRtOutputAllocatorCallback callback,
        void* user_state) noexcept
        : line_(line),
          owner_id_(g_output_allocator_owner_next_id.fetch_add(1U)),
          callback_(callback),
          user_state_(user_state)
    {
        copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native output allocator callback owner created in detached state.");
    }

    OutputAllocatorCallbackOwner(const OutputAllocatorCallbackOwner&) = delete;
    OutputAllocatorCallbackOwner& operator=(const OutputAllocatorCallbackOwner&) = delete;
    OutputAllocatorCallbackOwner(OutputAllocatorCallbackOwner&&) = delete;
    OutputAllocatorCallbackOwner& operator=(OutputAllocatorCallbackOwner&&) = delete;

    ~OutputAllocatorCallbackOwner() noexcept override
    {
        detach_noexcept();
    }

    void* reallocateOutput(
        char const* tensor_name,
        void* current_memory,
        uint64_t size,
        uint64_t alignment) noexcept override
    {
        return reallocate_output(tensor_name, current_memory, size, alignment, false);
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    void* reallocateOutputAsync(
        char const* tensor_name,
        void* current_memory,
        uint64_t size,
        uint64_t alignment,
        cudaStream_t stream) noexcept override
    {
        return reallocate_output(tensor_name, current_memory, size, alignment, stream != nullptr);
    }
#endif

    void notifyShape(char const* tensor_name, nvinfer1::Dims const& shape) noexcept override
    {
        if (!enter_callback())
        {
            return;
        }

        struct CallbackExit final
        {
            explicit CallbackExit(OutputAllocatorCallbackOwner* owner) noexcept : owner_(owner) {}
            ~CallbackExit() noexcept { owner_->leave_callback(); }
            OutputAllocatorCallbackOwner* owner_;
        } callback_exit(this);

        try
        {
            std::array<int64_t, 8> copied_shape{};
            const int32_t shape_rank = shape.nbDims >= 0 && shape.nbDims <= 8 ? shape.nbDims : 0;
            for (int32_t index = 0; index < shape_rank; ++index)
            {
                copied_shape[static_cast<size_t>(index)] = static_cast<int64_t>(shape.d[index]);
            }

            JYPPX_Boolean ignored_should_allocate = JYPPX_TRUE;
            const auto callback_status = invoke_callback(
                JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_NOTIFY_SHAPE,
                tensor_name,
                0U,
                0U,
                false,
                false,
                shape_rank,
                copied_shape,
                &ignored_should_allocate);

            notify_shape_count_.fetch_add(1U);
            if (callback_status != JYPPX_STATUS_OK)
            {
                failure_count_.fetch_add(1U);
            }
            record_request(
                JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_NOTIFY_SHAPE,
                tensor_name,
                0U,
                0U,
                false,
                false,
                shape_rank,
                copied_shape,
                callback_status,
                callback_status == JYPPX_STATUS_OK,
                "TensorRT invoked IOutputAllocator::notifyShape; copied shape metadata reached the managed callback.");
        }
        catch (const std::exception& exception)
        {
            record_failure(exception.what());
        }
        catch (...)
        {
            record_failure("unknown native exception in IOutputAllocator::notifyShape");
        }
    }

    JYPPX_StatusCode attach(
        nvinfer1::IExecutionContext* context,
        const char* tensor_name,
        JYPPX_Boolean* out_attached)
    {
        if (tensor_name == nullptr || tensor_name[0] == '\0')
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Output allocator tensor name must not be empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        if (attached_context_ != nullptr &&
            (attached_context_ != context || attached_tensor_name_ != tensor_name))
        {
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Output allocator owner is already attached to another execution context or tensor.");
            last_status_ = JYPPX_STATUS_INVALID_STATE;
            return JYPPX_STATUS_INVALID_STATE;
        }

        if (attached_context_ == context && attached_tensor_name_ == tensor_name)
        {
            detaching_ = false;
            *out_attached = JYPPX_TRUE;
            return JYPPX_STATUS_OK;
        }

        JYPPX_Boolean attached = JYPPX_FALSE;
        const auto status = set_output_allocator_with_seh_guard(
            context,
            tensor_name,
            this,
            &attached,
            "execution context output allocator owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            last_status_ = status;
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "TensorRT output allocator attach raised a guarded native failure.");
            return status;
        }

        if (attached == JYPPX_FALSE)
        {
            last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "IExecutionContext::setOutputAllocator returned false.");
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
            return JYPPX_STATUS_RUNTIME_ERROR;
        }

        attached_context_ = context;
        attached_tensor_name_ = tensor_name;
        detaching_ = false;
        attach_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_OK;
        copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native output allocator vtable attached to the TensorRT execution context and tensor.");
        *out_attached = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
    }

    JYPPX_StatusCode detach(JYPPX_Boolean* out_detached)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        detaching_ = true;
        if (attached_context_ != nullptr)
        {
            nvinfer1::IOutputAllocator* active_allocator = nullptr;
            auto status = get_output_allocator_with_seh_guard(
                attached_context_,
                attached_tensor_name_.c_str(),
                &active_allocator,
                "execution context output allocator owner detach query");
            if (status != JYPPX_STATUS_OK)
            {
                detaching_ = false;
                last_status_ = status;
                return status;
            }

            if (active_allocator == this)
            {
                JYPPX_Boolean cleared = JYPPX_FALSE;
                status = set_output_allocator_with_seh_guard(
                    attached_context_,
                    attached_tensor_name_.c_str(),
                    nullptr,
                    &cleared,
                    "execution context output allocator owner detach");
                if (status != JYPPX_STATUS_OK)
                {
                    detaching_ = false;
                    last_status_ = status;
                    return status;
                }

                if (cleared == JYPPX_FALSE)
                {
                    detaching_ = false;
                    last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
                    copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "IExecutionContext::setOutputAllocator(nullptr) returned false.");
                    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
                    return JYPPX_STATUS_RUNTIME_ERROR;
                }
            }

            attached_context_ = nullptr;
            attached_tensor_name_.clear();
            detach_count_.fetch_add(1U);
        }

        *out_detached = JYPPX_TRUE;
        callback_drained_.wait(lock, [this]() noexcept { return in_flight_callback_count_.load() == 0U; });
        release_allocations_after_synchronization_locked();
        return JYPPX_STATUS_OK;
    }

    void copy_info(JYPPX_TensorRtOutputAllocatorOwnerInfo* out_info) const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        out_info->line = static_cast<uint32_t>(line_);
        out_info->owner_id = owner_id_;
        out_info->invocation_count = invocation_count_.load();
        out_info->notify_shape_count = notify_shape_count_.load();
        out_info->reallocate_output_count = reallocate_output_count_.load();
        out_info->failure_count = failure_count_.load();
        out_info->in_flight_callback_count = in_flight_callback_count_.load();
        out_info->max_in_flight_callback_count = max_in_flight_callback_count_.load();
        out_info->attach_count = attach_count_.load();
        out_info->detach_count = detach_count_.load();
        out_info->allocation_count = allocation_count_.load();
        out_info->reuse_count = reuse_count_.load();
        out_info->release_count = release_count_.load();
        out_info->live_allocation_count = static_cast<uint64_t>(allocations_.size());
        out_info->live_allocation_bytes = live_allocation_bytes_;
        out_info->peak_live_allocation_bytes = peak_live_allocation_bytes_;
        out_info->last_requested_size = last_requested_size_;
        out_info->last_alignment = last_alignment_;
        out_info->last_status = static_cast<int32_t>(last_status_);
        out_info->is_attached = attached_context_ != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_callback_succeeded = last_callback_succeeded_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_allocation_succeeded = last_allocation_succeeded_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_had_current_memory = last_had_current_memory_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_had_stream = last_had_stream_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_callback_kind = last_callback_kind_;
        out_info->last_shape_rank = last_shape_rank_;
        for (size_t index = 0U; index < last_shape_.size(); ++index)
        {
            out_info->last_shape[index] = last_shape_[index];
        }

        copy_output_allocator_text(out_info->last_tensor_name, sizeof(out_info->last_tensor_name), last_tensor_name_);
        copy_output_allocator_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic_);
    }

private:
    struct AllocationRecord
    {
        void* base_pointer;
        void* aligned_pointer;
        uint64_t capacity;
        uint64_t allocated_bytes;
    };

    void* reallocate_output(
        const char* tensor_name,
        void* current_memory,
        const uint64_t size,
        const uint64_t alignment,
        const bool has_stream) noexcept
    {
        if (!enter_callback())
        {
            return nullptr;
        }

        struct CallbackExit final
        {
            explicit CallbackExit(OutputAllocatorCallbackOwner* owner) noexcept : owner_(owner) {}
            ~CallbackExit() noexcept { owner_->leave_callback(); }
            OutputAllocatorCallbackOwner* owner_;
        } callback_exit(this);

        try
        {
            std::array<int64_t, 8> empty_shape{};
            JYPPX_Boolean should_allocate = JYPPX_TRUE;
            const auto callback_status = invoke_callback(
                JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT,
                tensor_name,
                size,
                alignment,
                current_memory != nullptr,
                has_stream,
                0,
                empty_shape,
                &should_allocate);

            reallocate_output_count_.fetch_add(1U);
            if (callback_status != JYPPX_STATUS_OK || should_allocate == JYPPX_FALSE)
            {
                const auto status = callback_status != JYPPX_STATUS_OK ? callback_status : JYPPX_STATUS_INVALID_STATE;
                record_request(
                    JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT,
                    tensor_name,
                    size,
                    alignment,
                    current_memory != nullptr,
                    has_stream,
                    0,
                    empty_shape,
                    status,
                    callback_status == JYPPX_STATUS_OK,
                    callback_status != JYPPX_STATUS_OK
                        ? "Managed output allocator callback returned a failure status."
                        : "Managed output allocator callback rejected the native allocation request.");
                failure_count_.fetch_add(1U);
                return nullptr;
            }

            return allocate_or_reuse(tensor_name, current_memory, size, alignment, has_stream);
        }
        catch (const std::exception& exception)
        {
            record_failure(exception.what());
            return nullptr;
        }
        catch (...)
        {
            record_failure("unknown native exception in IOutputAllocator::reallocateOutput");
            return nullptr;
        }
    }

    void* allocate_or_reuse(
        const char* tensor_name,
        void* current_memory,
        const uint64_t size,
        const uint64_t alignment,
        const bool has_stream)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        const uint64_t safe_alignment = alignment == 0U ? 1U : alignment;
        auto current = allocations_.find(current_memory);
        if (current != allocations_.end() &&
            current->second.capacity >= size &&
            (reinterpret_cast<uintptr_t>(current_memory) % safe_alignment) == 0U)
        {
            reuse_count_.fetch_add(1U);
            record_request_locked(
                JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT,
                tensor_name,
                size,
                alignment,
                current_memory != nullptr,
                has_stream,
                0,
                std::array<int64_t, 8>{},
                JYPPX_STATUS_OK,
                true,
                true,
                "Native output allocator reused its owner-tracked CUDA allocation.");
            return current_memory;
        }

        if (safe_alignment > static_cast<uint64_t>((std::numeric_limits<uintptr_t>::max)()) ||
            size > static_cast<uint64_t>((std::numeric_limits<size_t>::max)()) ||
            size > (std::numeric_limits<uint64_t>::max)() - (safe_alignment - 1U))
        {
            record_allocation_failure_locked(tensor_name, size, alignment, current_memory != nullptr, has_stream, "Output allocation size or alignment exceeds the native address space.");
            return nullptr;
        }

        const uint64_t allocation_size = size + safe_alignment - 1U;
        if (allocation_size > static_cast<uint64_t>((std::numeric_limits<size_t>::max)()))
        {
            record_allocation_failure_locked(tensor_name, size, alignment, current_memory != nullptr, has_stream, "Output allocation byte count exceeds size_t.");
            return nullptr;
        }

        void* base_pointer = nullptr;
        const cudaError_t allocation_status = cudaMalloc(&base_pointer, static_cast<size_t>(allocation_size));
        if (allocation_status != cudaSuccess || base_pointer == nullptr)
        {
            record_allocation_failure_locked(tensor_name, size, alignment, current_memory != nullptr, has_stream, cudaGetErrorString(allocation_status));
            return nullptr;
        }

        const uintptr_t base_address = reinterpret_cast<uintptr_t>(base_pointer);
        const uintptr_t remainder = base_address % static_cast<uintptr_t>(safe_alignment);
        const uintptr_t aligned_address = remainder == 0U
            ? base_address
            : base_address + static_cast<uintptr_t>(safe_alignment) - remainder;
        void* aligned_pointer = reinterpret_cast<void*>(aligned_address);
        if (current != allocations_.end())
        {
            release_allocation_locked(current);
        }

        allocations_.emplace(
            aligned_pointer,
            AllocationRecord{base_pointer, aligned_pointer, size, allocation_size});
        live_allocation_bytes_ += allocation_size;
        if (live_allocation_bytes_ > peak_live_allocation_bytes_)
        {
            peak_live_allocation_bytes_ = live_allocation_bytes_;
        }

        allocation_count_.fetch_add(1U);

        record_request_locked(
            JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT,
            tensor_name,
            size,
            alignment,
            current_memory != nullptr,
            has_stream,
            0,
            std::array<int64_t, 8>{},
            JYPPX_STATUS_OK,
            true,
            true,
            "Native output allocator created an aligned owner-tracked CUDA allocation.");
        return aligned_pointer;
    }

    JYPPX_StatusCode invoke_callback(
        const int32_t callback_kind,
        const char* tensor_name,
        const uint64_t requested_size,
        const uint64_t alignment,
        const bool has_current_memory,
        const bool has_stream,
        const int32_t shape_rank,
        const std::array<int64_t, 8>& shape,
        JYPPX_Boolean* out_should_allocate) const noexcept
    {
        if (callback_ == nullptr)
        {
            return JYPPX_STATUS_INVALID_STATE;
        }

        const char* safe_name = tensor_name != nullptr ? tensor_name : "";
        return callback_(
            static_cast<uint32_t>(line_),
            callback_kind,
            safe_name,
            std::strlen(safe_name),
            requested_size,
            alignment,
            has_current_memory ? JYPPX_TRUE : JYPPX_FALSE,
            has_stream ? JYPPX_TRUE : JYPPX_FALSE,
            shape_rank,
            shape[0], shape[1], shape[2], shape[3],
            shape[4], shape[5], shape[6], shape[7],
            out_should_allocate,
            user_state_);
    }

    void record_request(
        const int32_t callback_kind,
        const char* tensor_name,
        const uint64_t requested_size,
        const uint64_t alignment,
        const bool has_current_memory,
        const bool has_stream,
        const int32_t shape_rank,
        const std::array<int64_t, 8>& shape,
        const JYPPX_StatusCode status,
        const bool callback_succeeded,
        const char* diagnostic)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        record_request_locked(
            callback_kind,
            tensor_name,
            requested_size,
            alignment,
            has_current_memory,
            has_stream,
            shape_rank,
            shape,
            status,
            callback_succeeded,
            last_allocation_succeeded_,
            diagnostic);
    }

    void record_request_locked(
        const int32_t callback_kind,
        const char* tensor_name,
        const uint64_t requested_size,
        const uint64_t alignment,
        const bool has_current_memory,
        const bool has_stream,
        const int32_t shape_rank,
        const std::array<int64_t, 8>& shape,
        const JYPPX_StatusCode status,
        const bool callback_succeeded,
        const bool allocation_succeeded,
        const char* diagnostic)
    {
        invocation_count_.fetch_add(1U);
        last_callback_kind_ = callback_kind;
        last_requested_size_ = requested_size;
        last_alignment_ = alignment;
        last_had_current_memory_ = has_current_memory;
        last_had_stream_ = has_stream;
        last_shape_rank_ = shape_rank;
        last_shape_ = shape;
        last_status_ = status;
        last_callback_succeeded_ = callback_succeeded;
        last_allocation_succeeded_ = allocation_succeeded;
        copy_output_allocator_text(last_tensor_name_, sizeof(last_tensor_name_), tensor_name);
        copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
        }
    }

    void record_allocation_failure_locked(
        const char* tensor_name,
        const uint64_t size,
        const uint64_t alignment,
        const bool has_current_memory,
        const bool has_stream,
        const char* detail)
    {
        failure_count_.fetch_add(1U);
        std::string diagnostic = "Native output allocator CUDA allocation failed: ";
        diagnostic += detail != nullptr ? detail : "unknown CUDA error";
        record_request_locked(
            JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_REALLOCATE_OUTPUT,
            tensor_name,
            size,
            alignment,
            has_current_memory,
            has_stream,
            0,
            std::array<int64_t, 8>{},
            JYPPX_STATUS_RUNTIME_ERROR,
            true,
            false,
            diagnostic.c_str());
    }

    void record_failure(const char* detail) noexcept
    {
        failure_count_.fetch_add(1U);
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
            last_callback_succeeded_ = false;
            last_allocation_succeeded_ = false;
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), detail);
        }
        catch (...)
        {
        }
    }

    bool enter_callback() noexcept
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (detaching_)
        {
            return false;
        }

        const uint64_t current = in_flight_callback_count_.fetch_add(1U) + 1U;
        uint64_t observed = max_in_flight_callback_count_.load();
        while (current > observed && !max_in_flight_callback_count_.compare_exchange_weak(observed, current))
        {
        }

        return true;
    }

    void leave_callback() noexcept
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        in_flight_callback_count_.fetch_sub(1U);
        callback_drained_.notify_all();
    }

    void release_allocation_locked(std::unordered_map<void*, AllocationRecord>::iterator allocation) noexcept
    {
        const cudaError_t status = cudaFree(allocation->second.base_pointer);
        if (status == cudaSuccess)
        {
            live_allocation_bytes_ -= allocation->second.allocated_bytes;
            release_count_.fetch_add(1U);
            allocations_.erase(allocation);
            return;
        }

        failure_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
        std::string diagnostic = "Native output allocator could not release CUDA memory: ";
        diagnostic += cudaGetErrorString(status);
        copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic.c_str());
    }

    void release_allocations_after_synchronization_locked() noexcept
    {
        if (allocations_.empty())
        {
            last_status_ = JYPPX_STATUS_OK;
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native output allocator detached with no live CUDA allocations.");
            return;
        }

        const cudaError_t synchronize_status = cudaDeviceSynchronize();
        if (synchronize_status != cudaSuccess)
        {
            failure_count_.fetch_add(1U);
            last_status_ = JYPPX_STATUS_RUNTIME_ERROR;
            std::string diagnostic = "Output allocator detached, but CUDA synchronization failed; allocations were intentionally retained: ";
            diagnostic += cudaGetErrorString(synchronize_status);
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic.c_str());
            return;
        }

        for (auto allocation = allocations_.begin(); allocation != allocations_.end();)
        {
            auto current = allocation++;
            release_allocation_locked(current);
        }

        if (allocations_.empty())
        {
            last_status_ = JYPPX_STATUS_OK;
            copy_output_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native output allocator detached after CUDA work completed and released all owned allocations.");
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

public:
    bool prepare_for_destroy_noexcept() noexcept
    {
        try
        {
            JYPPX_Boolean detached = JYPPX_FALSE;
            return detach(&detached) == JYPPX_STATUS_OK && detached == JYPPX_TRUE;
        }
        catch (...)
        {
            return false;
        }
    }

private:
    JYPPX_TensorRtLine line_;
    uint64_t owner_id_;
    JYPPX_TensorRtOutputAllocatorCallback callback_;
    void* user_state_;
    mutable std::mutex state_mutex_;
    std::condition_variable callback_drained_;
    nvinfer1::IExecutionContext* attached_context_{nullptr};
    std::string attached_tensor_name_;
    bool detaching_{false};
    std::unordered_map<void*, AllocationRecord> allocations_;
    std::atomic<uint64_t> invocation_count_{0U};
    std::atomic<uint64_t> notify_shape_count_{0U};
    std::atomic<uint64_t> reallocate_output_count_{0U};
    std::atomic<uint64_t> failure_count_{0U};
    std::atomic<uint64_t> in_flight_callback_count_{0U};
    std::atomic<uint64_t> max_in_flight_callback_count_{0U};
    std::atomic<uint64_t> attach_count_{0U};
    std::atomic<uint64_t> detach_count_{0U};
    std::atomic<uint64_t> allocation_count_{0U};
    std::atomic<uint64_t> reuse_count_{0U};
    std::atomic<uint64_t> release_count_{0U};
    uint64_t live_allocation_bytes_{0U};
    uint64_t peak_live_allocation_bytes_{0U};
    uint64_t last_requested_size_{0U};
    uint64_t last_alignment_{0U};
    JYPPX_StatusCode last_status_{JYPPX_STATUS_OK};
    bool last_callback_succeeded_{false};
    bool last_allocation_succeeded_{false};
    bool last_had_current_memory_{false};
    bool last_had_stream_{false};
    int32_t last_callback_kind_{JYPPX_TENSORRT_OUTPUT_ALLOCATOR_CALLBACK_UNKNOWN};
    int32_t last_shape_rank_{0};
    std::array<int64_t, 8> last_shape_{};
    char last_tensor_name_[256]{};
    char last_diagnostic_[1024]{};
};

static_assert(
    !std::is_copy_constructible<OutputAllocatorCallbackOwner>::value &&
        !std::is_copy_assignable<OutputAllocatorCallbackOwner>::value,
    "Output allocator callback owners must not be copyable.");
static_assert(
    !std::is_move_constructible<OutputAllocatorCallbackOwner>::value &&
        !std::is_move_assignable<OutputAllocatorCallbackOwner>::value,
    "Output allocator callback owners must not be movable.");
static_assert(
    std::is_nothrow_destructible<OutputAllocatorCallbackOwner>::value,
    "Output allocator callback owner destruction must not throw.");

static void destroy_output_allocator_callback_owner(void* payload) noexcept
{
    auto* owner = static_cast<OutputAllocatorCallbackOwner*>(payload);
    if (owner != nullptr && owner->prepare_for_destroy_noexcept())
    {
        delete owner;
    }
}

static JYPPX_StatusCode get_output_allocator_callback_owner(
    JYPPX_TensorRtOutputAllocatorOwner* owner,
    OutputAllocatorCallbackOwner** out_payload)
{
    auto status = jyppx::tensorrt::validate_handle(
        owner,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_OUTPUT_ALLOCATOR_CALLBACK_OWNER,
        "owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<OutputAllocatorCallbackOwner>(owner);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Output allocator owner handle has no callback payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API(output_allocator_owner_create)(
    JYPPX_TensorRtOutputAllocatorCallback callback,
    void* user_state,
    JYPPX_TensorRtOutputAllocatorOwner** out_owner)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_owner, "out_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_owner = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Output allocator callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        auto* payload = new (std::nothrow) OutputAllocatorCallbackOwner(kLine, callback, user_state);
        if (payload == nullptr)
        {
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_OUTPUT_ALLOCATOR_CALLBACK_OWNER,
            payload,
            &destroy_output_allocator_callback_owner);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        *out_owner = reinterpret_cast<JYPPX_TensorRtOutputAllocatorOwner*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "output allocator callback owner requires the CUDA toolkit");
#else
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, kOutputAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API(output_allocator_owner_attach)(
    JYPPX_TensorRtOutputAllocatorOwner* owner,
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_attached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attached, "out_attached");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_attached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        OutputAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_output_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        nvinfer1::IExecutionContext* context_payload = nullptr;
        status = get_context_payload_ext(context, &context_payload, "execution context output allocator owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        return owner_payload->attach(context_payload, tensor_name, out_attached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    (void)context;
    (void)tensor_name;
    return jyppx::tensorrt::report_vendor_missing(kLine, "output allocator callback owner requires the CUDA toolkit");
#else
    (void)owner;
    (void)context;
    (void)tensor_name;
    return jyppx::tensorrt::report_vendor_missing(kLine, kOutputAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API(output_allocator_owner_detach)(
    JYPPX_TensorRtOutputAllocatorOwner* owner,
    JYPPX_Boolean* out_detached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_detached, "out_detached");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_detached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        OutputAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_output_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        return owner_payload->detach(out_detached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, "output allocator callback owner requires the CUDA toolkit");
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kOutputAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_OUTPUT_ALLOCATOR_OWNER_API(output_allocator_owner_get_info)(
    JYPPX_TensorRtOutputAllocatorOwner* owner,
    JYPPX_TensorRtOutputAllocatorOwnerInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_info, 0, sizeof(*out_info));
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        OutputAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_output_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        owner_payload->copy_info(out_info);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kOutputAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, "output allocator callback owner requires the CUDA toolkit");
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kOutputAllocatorOwnerFeature);
#endif
}
