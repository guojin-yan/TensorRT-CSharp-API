#ifndef JYPPX_TRT_GPU_ALLOCATOR_OWNER_API
#error "JYPPX_TRT_GPU_ALLOCATOR_OWNER_API must be defined before including gpu_allocator_callback_owner.inc"
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>

#ifndef JYPPX_HAS_CUDA_TOOLKIT
#define JYPPX_HAS_CUDA_TOOLKIT 0
#endif

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace
{
constexpr const char* kGpuAllocatorOwnerFeature = "GPU allocator callback owner";
static std::atomic<uint64_t> g_gpu_allocator_owner_next_id{1U};

static void copy_gpu_allocator_text(char* destination, const size_t destination_size, const char* source) noexcept
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
static JYPPX_StatusCode set_runtime_gpu_allocator_with_seh_guard(
    nvinfer1::IRuntime* runtime,
    nvinfer1::IGpuAllocator* allocator,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        runtime->setGpuAllocator(allocator);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    runtime->setGpuAllocator(allocator);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_builder_gpu_allocator_with_seh_guard(
    nvinfer1::IBuilder* builder,
    nvinfer1::IGpuAllocator* allocator,
    const char* feature_name) noexcept
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        builder->setGpuAllocator(allocator);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    builder->setGpuAllocator(allocator);
    return JYPPX_STATUS_OK;
#endif
}

class GpuAllocatorCallbackOwner final : public nvinfer1::IGpuAllocator
{
public:
    GpuAllocatorCallbackOwner(
        const JYPPX_TensorRtLine line,
        JYPPX_TensorRtGpuAllocatorCallback callback,
        void* user_state) noexcept
        : line_(line),
          owner_id_(g_gpu_allocator_owner_next_id.fetch_add(1U)),
          callback_(callback),
          user_state_(user_state)
    {
        copy_gpu_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native GPU allocator owner created in detached state.");
    }

    GpuAllocatorCallbackOwner(const GpuAllocatorCallbackOwner&) = delete;
    GpuAllocatorCallbackOwner& operator=(const GpuAllocatorCallbackOwner&) = delete;
    GpuAllocatorCallbackOwner(GpuAllocatorCallbackOwner&&) = delete;
    GpuAllocatorCallbackOwner& operator=(GpuAllocatorCallbackOwner&&) = delete;

    ~GpuAllocatorCallbackOwner() noexcept override
    {
        detach_noexcept();
        release_remaining_allocations_noexcept();
    }

    void* allocate(
        const uint64_t size,
        const uint64_t alignment,
        const nvinfer1::AllocatorFlags flags) noexcept override
    {
        return allocate_impl(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE,
            size,
            alignment,
            static_cast<uint32_t>(flags),
            false);
    }

    void* reallocate(void* const base_address, const uint64_t alignment, const uint64_t new_size) noexcept override
    {
        CallbackResult callback = invoke_callback(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_REALLOCATE,
            new_size,
            alignment,
            0U,
            base_address != nullptr,
            false);
        if (!callback.proceed)
        {
            record_operation(false, callback.status, "GPU reallocation was rejected by the managed policy callback.");
            return nullptr;
        }

        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (base_address == nullptr)
            {
                return allocate_cuda_locked(new_size, alignment, "GPU reallocation created a new native CUDA allocation.");
            }

            auto existing = allocations_.find(base_address);
            if (existing == allocations_.end())
            {
                cuda_failure_count_.fetch_add(1U);
                record_operation_locked(false, JYPPX_STATUS_INVALID_ARGUMENT, "GPU reallocation received memory not owned by this allocator.");
                return nullptr;
            }

            if (new_size == 0U)
            {
                if (!release_allocation_locked(existing))
                {
                    return nullptr;
                }

                record_operation_locked(true, JYPPX_STATUS_OK, "GPU reallocation released the allocation for a zero-size request.");
                return nullptr;
            }

            void* replacement = nullptr;
            const cudaError_t allocation_status = cudaMalloc(&replacement, static_cast<size_t>(new_size));
            if (allocation_status != cudaSuccess || replacement == nullptr)
            {
                record_cuda_failure_locked("cudaMalloc failed during GPU reallocation", allocation_status);
                return nullptr;
            }

            const size_t copy_size = static_cast<size_t>(std::min(existing->second.size, new_size));
            const cudaError_t copy_status = cudaMemcpy(replacement, base_address, copy_size, cudaMemcpyDeviceToDevice);
            if (copy_status != cudaSuccess)
            {
                (void)cudaFree(replacement);
                record_cuda_failure_locked("cudaMemcpy failed during GPU reallocation", copy_status);
                return nullptr;
            }

            const uint64_t previous_size = existing->second.size;
            const cudaError_t release_status = cudaFree(base_address);
            if (release_status != cudaSuccess)
            {
                (void)cudaFree(replacement);
                record_cuda_failure_locked("cudaFree failed during GPU reallocation", release_status);
                return nullptr;
            }

            allocations_.erase(existing);
            allocations_.emplace(replacement, AllocationRecord{new_size});
            live_allocation_bytes_ = live_allocation_bytes_ - previous_size + new_size;
            peak_live_allocation_bytes_ = std::max(peak_live_allocation_bytes_, live_allocation_bytes_);
            record_operation_locked(true, JYPPX_STATUS_OK, "GPU reallocation moved data into a new owner-tracked CUDA allocation.");
            return replacement;
        }
        catch (...)
        {
            record_native_exception("Native exception escaped the GPU reallocation implementation.");
            return nullptr;
        }
    }

    bool deallocate(void* const memory) noexcept override
    {
        return deallocate_impl(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE,
            memory,
            false);
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM < 10
    void free(void* const memory) noexcept override
    {
        // TensorRT 8 keeps free() as a pure virtual compatibility entry point.
        (void)deallocate_impl(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE,
            memory,
            false);
    }
#endif

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    void* allocateAsync(
        const uint64_t size,
        const uint64_t alignment,
        const nvinfer1::AllocatorFlags flags,
        cudaStream_t) noexcept override
    {
        // A synchronous implementation is explicitly permitted by IGpuAllocator.
        return allocate_impl(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE_ASYNC,
            size,
            alignment,
            static_cast<uint32_t>(flags),
            true);
    }

    bool deallocateAsync(void* const memory, cudaStream_t) noexcept override
    {
        // cudaFree may synchronize, which is valid for IGpuAllocator::deallocateAsync.
        return deallocate_impl(
            JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE_ASYNC,
            memory,
            true);
    }
#endif

    JYPPX_StatusCode attach_runtime(nvinfer1::IRuntime* runtime, JYPPX_Boolean* out_attached)
    {
        if (runtime == nullptr)
        {
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (attachment_target_ != JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE)
            {
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "GPU allocator owner is already attached.");
                return JYPPX_STATUS_INVALID_STATE;
            }
        }

        const auto status = set_runtime_gpu_allocator_with_seh_guard(runtime, this, "runtime GPU allocator owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        attached_runtime_ = runtime;
        attachment_target_ = JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_RUNTIME;
        attach_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_OK;
        copy_gpu_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native GPU allocator owner attached to a TensorRT runtime.");
        *out_attached = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
    }

    JYPPX_StatusCode attach_builder(nvinfer1::IBuilder* builder, JYPPX_Boolean* out_attached)
    {
        if (builder == nullptr)
        {
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (attachment_target_ != JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE)
            {
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "GPU allocator owner is already attached.");
                return JYPPX_STATUS_INVALID_STATE;
            }
        }

        const auto status = set_builder_gpu_allocator_with_seh_guard(builder, this, "builder GPU allocator owner attach");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        attached_builder_ = builder;
        attachment_target_ = JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_BUILDER;
        attach_count_.fetch_add(1U);
        last_status_ = JYPPX_STATUS_OK;
        copy_gpu_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), "Native GPU allocator owner attached to a TensorRT builder.");
        *out_attached = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
    }

    JYPPX_StatusCode detach(JYPPX_Boolean* out_detached) noexcept
    {
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IBuilder* builder = nullptr;
        int32_t target = JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            runtime = attached_runtime_;
            builder = attached_builder_;
            target = attachment_target_;
        }

        JYPPX_StatusCode status = JYPPX_STATUS_OK;
        if (target == JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_RUNTIME)
        {
            status = set_runtime_gpu_allocator_with_seh_guard(runtime, nullptr, "runtime GPU allocator owner detach");
        }
        else if (target == JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_BUILDER)
        {
            status = set_builder_gpu_allocator_with_seh_guard(builder, nullptr, "builder GPU allocator owner detach");
        }

        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::unique_lock<std::mutex> lock(state_mutex_);
        callback_drained_.wait(lock, [this] { return in_flight_callback_count_.load() == 0U; });
        attached_runtime_ = nullptr;
        attached_builder_ = nullptr;
        attachment_target_ = JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE;
        if (target != JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE)
        {
            detach_count_.fetch_add(1U);
        }
        last_status_ = JYPPX_STATUS_OK;
        copy_gpu_allocator_text(
            last_diagnostic_,
            sizeof(last_diagnostic_),
            target == JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE
                ? "Native GPU allocator owner was already detached."
                : "Native GPU allocator owner detached; live allocations remain owned until TensorRT releases them.");
        *out_detached = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
    }

    void copy_info(JYPPX_TensorRtGpuAllocatorOwnerInfo* out_info) const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        out_info->line = static_cast<uint32_t>(line_);
        out_info->owner_id = owner_id_;
        out_info->invocation_count = invocation_count_.load();
        out_info->allocate_count = allocate_count_.load();
        out_info->reallocate_count = reallocate_count_.load();
        out_info->deallocate_count = deallocate_count_.load();
        out_info->allocate_async_count = allocate_async_count_.load();
        out_info->deallocate_async_count = deallocate_async_count_.load();
        out_info->rejected_count = rejected_count_.load();
        out_info->callback_failure_count = callback_failure_count_.load();
        out_info->cuda_failure_count = cuda_failure_count_.load();
        out_info->in_flight_callback_count = in_flight_callback_count_.load();
        out_info->max_in_flight_callback_count = max_in_flight_callback_count_.load();
        out_info->attach_count = attach_count_.load();
        out_info->detach_count = detach_count_.load();
        out_info->live_allocation_count = static_cast<uint64_t>(allocations_.size());
        out_info->live_allocation_bytes = live_allocation_bytes_;
        out_info->peak_live_allocation_bytes = peak_live_allocation_bytes_;
        out_info->last_requested_size = last_requested_size_;
        out_info->last_alignment = last_alignment_;
        out_info->last_allocator_flags = last_allocator_flags_;
        out_info->last_status = static_cast<int32_t>(last_status_);
        out_info->attachment_target = attachment_target_;
        out_info->last_callback_kind = last_callback_kind_;
        out_info->is_attached = attachment_target_ != JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_callback_succeeded = last_callback_succeeded_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_operation_succeeded = last_operation_succeeded_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_had_current_memory = last_had_current_memory_ ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->last_had_stream = last_had_stream_ ? JYPPX_TRUE : JYPPX_FALSE;
        copy_gpu_allocator_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic_);
    }

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
    struct AllocationRecord
    {
        uint64_t size;
    };

    struct CallbackResult
    {
        JYPPX_StatusCode status;
        bool proceed;
    };

    void* allocate_impl(
        const int32_t kind,
        const uint64_t size,
        const uint64_t alignment,
        const uint32_t flags,
        const bool has_stream) noexcept
    {
        CallbackResult callback = invoke_callback(kind, size, alignment, flags, false, has_stream);
        if (!callback.proceed)
        {
            record_operation(false, callback.status, "GPU allocation was rejected by the managed policy callback.");
            return nullptr;
        }

        if (size == 0U)
        {
            record_operation(true, JYPPX_STATUS_OK, "GPU allocator accepted a zero-size request and returned null as required.");
            return nullptr;
        }

        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return allocate_cuda_locked(size, alignment, "GPU allocator created an owner-tracked CUDA allocation.");
        }
        catch (...)
        {
            record_native_exception("Native exception escaped the GPU allocation implementation.");
            return nullptr;
        }
    }

    bool deallocate_impl(const int32_t kind, void* memory, const bool has_stream) noexcept
    {
        CallbackResult callback = invoke_callback(kind, 0U, 0U, 0U, memory != nullptr, has_stream);
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (memory == nullptr)
            {
                record_operation_locked(true, callback.status, "GPU allocator accepted a null deallocation request.");
                return true;
            }

            auto allocation = allocations_.find(memory);
            if (allocation == allocations_.end())
            {
                cuda_failure_count_.fetch_add(1U);
                record_operation_locked(false, JYPPX_STATUS_INVALID_ARGUMENT, "GPU deallocation received memory not owned by this allocator.");
                return false;
            }

            const bool released = release_allocation_locked(allocation);
            if (released && callback.status != JYPPX_STATUS_OK)
            {
                record_operation_locked(true, callback.status, "GPU memory was released after the managed observation callback failed.");
            }
            else if (released)
            {
                record_operation_locked(true, JYPPX_STATUS_OK, "GPU allocator released an owner-tracked CUDA allocation.");
            }
            return released;
        }
        catch (...)
        {
            record_native_exception("Native exception escaped the GPU deallocation implementation.");
            return false;
        }
    }

    void* allocate_cuda_locked(const uint64_t size, const uint64_t, const char* diagnostic)
    {
        void* memory = nullptr;
        const cudaError_t status = cudaMalloc(&memory, static_cast<size_t>(size));
        if (status != cudaSuccess || memory == nullptr)
        {
            record_cuda_failure_locked("cudaMalloc failed in the GPU allocator owner", status);
            return nullptr;
        }

        allocations_.emplace(memory, AllocationRecord{size});
        live_allocation_bytes_ += size;
        peak_live_allocation_bytes_ = std::max(peak_live_allocation_bytes_, live_allocation_bytes_);
        record_operation_locked(true, JYPPX_STATUS_OK, diagnostic);
        return memory;
    }

    bool release_allocation_locked(std::unordered_map<void*, AllocationRecord>::iterator allocation)
    {
        const cudaError_t status = cudaFree(allocation->first);
        if (status != cudaSuccess)
        {
            record_cuda_failure_locked("cudaFree failed in the GPU allocator owner", status);
            return false;
        }

        live_allocation_bytes_ -= allocation->second.size;
        allocations_.erase(allocation);
        return true;
    }

    CallbackResult invoke_callback(
        const int32_t kind,
        const uint64_t size,
        const uint64_t alignment,
        const uint32_t flags,
        const bool has_current_memory,
        const bool has_stream) noexcept
    {
        const uint64_t current = in_flight_callback_count_.fetch_add(1U) + 1U;
        uint64_t observed = max_in_flight_callback_count_.load();
        while (current > observed && !max_in_flight_callback_count_.compare_exchange_weak(observed, current))
        {
        }

        JYPPX_Boolean should_proceed = JYPPX_FALSE;
        JYPPX_StatusCode status = JYPPX_STATUS_INVALID_STATE;
        if (callback_ != nullptr)
        {
            status = callback_(
                static_cast<uint32_t>(line_),
                kind,
                size,
                alignment,
                flags,
                has_current_memory ? JYPPX_TRUE : JYPPX_FALSE,
                has_stream ? JYPPX_TRUE : JYPPX_FALSE,
                &should_proceed,
                user_state_);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            invocation_count_.fetch_add(1U);
            increment_kind_count(kind);
            last_callback_kind_ = kind;
            last_requested_size_ = size;
            last_alignment_ = alignment;
            last_allocator_flags_ = flags;
            last_had_current_memory_ = has_current_memory;
            last_had_stream_ = has_stream;
            last_callback_succeeded_ = status == JYPPX_STATUS_OK;
            if (status != JYPPX_STATUS_OK)
            {
                callback_failure_count_.fetch_add(1U);
            }
            else if (should_proceed == JYPPX_FALSE &&
                kind != JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE &&
                kind != JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE_ASYNC)
            {
                rejected_count_.fetch_add(1U);
            }
        }

        in_flight_callback_count_.fetch_sub(1U);
        callback_drained_.notify_all();
        const bool deallocation = kind == JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE ||
            kind == JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE_ASYNC;
        return CallbackResult{status, deallocation || (status == JYPPX_STATUS_OK && should_proceed == JYPPX_TRUE)};
    }

    void increment_kind_count(const int32_t kind) noexcept
    {
        switch (kind)
        {
        case JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE:
            allocate_count_.fetch_add(1U);
            break;
        case JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_REALLOCATE:
            reallocate_count_.fetch_add(1U);
            break;
        case JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE:
            deallocate_count_.fetch_add(1U);
            break;
        case JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_ALLOCATE_ASYNC:
            allocate_async_count_.fetch_add(1U);
            break;
        case JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_DEALLOCATE_ASYNC:
            deallocate_async_count_.fetch_add(1U);
            break;
        default:
            break;
        }
    }

    void record_operation(const bool succeeded, const JYPPX_StatusCode status, const char* diagnostic) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            record_operation_locked(succeeded, status, diagnostic);
        }
        catch (...)
        {
        }
    }

    void record_operation_locked(const bool succeeded, const JYPPX_StatusCode status, const char* diagnostic) noexcept
    {
        last_operation_succeeded_ = succeeded;
        last_status_ = status;
        copy_gpu_allocator_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, last_diagnostic_);
        }
    }

    void record_cuda_failure_locked(const char* prefix, const cudaError_t status) noexcept
    {
        cuda_failure_count_.fetch_add(1U);
        std::string diagnostic = prefix != nullptr ? prefix : "CUDA operation failed";
        diagnostic += ": ";
        diagnostic += cudaGetErrorString(status);
        record_operation_locked(false, JYPPX_STATUS_RUNTIME_ERROR, diagnostic.c_str());
    }

    void record_native_exception(const char* diagnostic) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            cuda_failure_count_.fetch_add(1U);
            record_operation_locked(false, JYPPX_STATUS_RUNTIME_ERROR, diagnostic);
        }
        catch (...)
        {
        }
    }

    void release_remaining_allocations_noexcept() noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (allocations_.empty())
            {
                return;
            }

            (void)cudaDeviceSynchronize();
            for (auto allocation = allocations_.begin(); allocation != allocations_.end();)
            {
                auto current = allocation++;
                (void)release_allocation_locked(current);
            }
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
    JYPPX_TensorRtGpuAllocatorCallback callback_;
    void* user_state_;
    mutable std::mutex state_mutex_;
    std::condition_variable callback_drained_;
    nvinfer1::IRuntime* attached_runtime_{nullptr};
    nvinfer1::IBuilder* attached_builder_{nullptr};
    int32_t attachment_target_{JYPPX_TENSORRT_GPU_ALLOCATOR_TARGET_NONE};
    std::unordered_map<void*, AllocationRecord> allocations_;
    std::atomic<uint64_t> invocation_count_{0U};
    std::atomic<uint64_t> allocate_count_{0U};
    std::atomic<uint64_t> reallocate_count_{0U};
    std::atomic<uint64_t> deallocate_count_{0U};
    std::atomic<uint64_t> allocate_async_count_{0U};
    std::atomic<uint64_t> deallocate_async_count_{0U};
    std::atomic<uint64_t> rejected_count_{0U};
    std::atomic<uint64_t> callback_failure_count_{0U};
    std::atomic<uint64_t> cuda_failure_count_{0U};
    std::atomic<uint64_t> in_flight_callback_count_{0U};
    std::atomic<uint64_t> max_in_flight_callback_count_{0U};
    std::atomic<uint64_t> attach_count_{0U};
    std::atomic<uint64_t> detach_count_{0U};
    uint64_t live_allocation_bytes_{0U};
    uint64_t peak_live_allocation_bytes_{0U};
    uint64_t last_requested_size_{0U};
    uint64_t last_alignment_{0U};
    uint32_t last_allocator_flags_{0U};
    JYPPX_StatusCode last_status_{JYPPX_STATUS_OK};
    int32_t last_callback_kind_{JYPPX_TENSORRT_GPU_ALLOCATOR_CALLBACK_UNKNOWN};
    bool last_callback_succeeded_{false};
    bool last_operation_succeeded_{false};
    bool last_had_current_memory_{false};
    bool last_had_stream_{false};
    char last_diagnostic_[1024]{};
};

static_assert(
    !std::is_copy_constructible<GpuAllocatorCallbackOwner>::value &&
        !std::is_copy_assignable<GpuAllocatorCallbackOwner>::value,
    "GPU allocator callback owners must not be copyable.");
static_assert(
    !std::is_move_constructible<GpuAllocatorCallbackOwner>::value &&
        !std::is_move_assignable<GpuAllocatorCallbackOwner>::value,
    "GPU allocator callback owners must not be movable.");
static_assert(
    std::is_nothrow_destructible<GpuAllocatorCallbackOwner>::value,
    "GPU allocator callback owner destruction must not throw.");

static void destroy_gpu_allocator_callback_owner(void* payload) noexcept
{
    auto* owner = static_cast<GpuAllocatorCallbackOwner*>(payload);
    if (owner != nullptr && owner->prepare_for_destroy_noexcept())
    {
        delete owner;
    }
}

static JYPPX_StatusCode get_gpu_allocator_callback_owner(
    JYPPX_TensorRtGpuAllocatorOwner* owner,
    GpuAllocatorCallbackOwner** out_payload)
{
    auto status = jyppx::tensorrt::validate_handle(
        owner,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_GPU_ALLOCATOR_CALLBACK_OWNER,
        "owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<GpuAllocatorCallbackOwner>(owner);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "GPU allocator owner handle has no callback payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_GPU_ALLOCATOR_OWNER_API(gpu_allocator_owner_create)(
    JYPPX_TensorRtGpuAllocatorCallback callback,
    void* user_state,
    JYPPX_TensorRtGpuAllocatorOwner** out_owner)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_owner, "out_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_owner = nullptr;
    if (callback == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "GPU allocator callback must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        auto* payload = new (std::nothrow) GpuAllocatorCallbackOwner(kLine, callback, user_state);
        if (payload == nullptr)
        {
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_GPU_ALLOCATOR_CALLBACK_OWNER,
            payload,
            &destroy_gpu_allocator_callback_owner);
        if (status != JYPPX_STATUS_OK)
        {
            delete payload;
            return status;
        }

        *out_owner = reinterpret_cast<JYPPX_TensorRtGpuAllocatorOwner*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "GPU allocator callback owner requires the CUDA toolkit");
#else
    (void)user_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, kGpuAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_GPU_ALLOCATOR_OWNER_API(gpu_allocator_owner_attach_runtime)(
    JYPPX_TensorRtGpuAllocatorOwner* owner,
    JYPPX_TensorRtRuntime* runtime,
    JYPPX_Boolean* out_attached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attached, "out_attached");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_attached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        GpuAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_gpu_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK) { return status; }
        status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
        if (status != JYPPX_STATUS_OK) { return status; }
        auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
        if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        return owner_payload->attach_runtime(runtime_payload, out_attached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner; (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "GPU allocator callback owner requires the CUDA toolkit");
#else
    (void)owner; (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, kGpuAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_GPU_ALLOCATOR_OWNER_API(gpu_allocator_owner_attach_builder)(
    JYPPX_TensorRtGpuAllocatorOwner* owner,
    JYPPX_TensorRtBuilder* builder,
    JYPPX_Boolean* out_attached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_attached, "out_attached");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_attached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        GpuAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_gpu_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK) { return status; }
        status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
        if (status != JYPPX_STATUS_OK) { return status; }
        auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
        if (builder_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        return owner_payload->attach_builder(builder_payload, out_attached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner; (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "GPU allocator callback owner requires the CUDA toolkit");
#else
    (void)owner; (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, kGpuAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_GPU_ALLOCATOR_OWNER_API(gpu_allocator_owner_detach)(
    JYPPX_TensorRtGpuAllocatorOwner* owner,
    JYPPX_Boolean* out_detached)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_detached, "out_detached");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_detached = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        GpuAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_gpu_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK) { return status; }
        return owner_payload->detach(out_detached);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, "GPU allocator callback owner requires the CUDA toolkit");
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kGpuAllocatorOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_GPU_ALLOCATOR_OWNER_API(gpu_allocator_owner_get_info)(
    JYPPX_TensorRtGpuAllocatorOwner* owner,
    JYPPX_TensorRtGpuAllocatorOwnerInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_info, 0, sizeof(*out_info));
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        GpuAllocatorCallbackOwner* owner_payload = nullptr;
        status = get_gpu_allocator_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK) { return status; }
        owner_payload->copy_info(out_info);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kGpuAllocatorOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, "GPU allocator callback owner requires the CUDA toolkit");
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kGpuAllocatorOwnerFeature);
#endif
}
