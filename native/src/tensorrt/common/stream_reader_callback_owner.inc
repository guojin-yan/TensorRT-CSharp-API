#ifndef JYPPX_TRT_STREAM_READER_OWNER_API
#error "JYPPX_TRT_STREAM_READER_OWNER_API must be defined before including stream_reader_callback_owner.inc"
#endif
#ifndef JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR
#error "JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR must be defined before including stream_reader_callback_owner.inc"
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef JYPPX_HAS_CUDA_TOOLKIT
#define JYPPX_HAS_CUDA_TOOLKIT 0
#endif

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace
{
constexpr const char* kStreamReaderOwnerFeature = "stream reader v2 callback owner";
static std::atomic<uint64_t> g_stream_reader_owner_next_id{1U};

static void copy_stream_reader_text(char* destination, const size_t destination_size, const char* source) noexcept
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

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR
static JYPPX_StatusCode deserialize_stream_reader_with_seh_guard(
    nvinfer1::IRuntime* runtime,
    nvinfer1::IStreamReaderV2* owner,
    nvinfer1::ICudaEngine** out_engine)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0U;
    __try
    {
        *out_engine = runtime->deserializeCudaEngine(*owner);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_engine = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kStreamReaderOwnerFeature, seh_exception_code);
    }
#else
    *out_engine = runtime->deserializeCudaEngine(*owner);
    return JYPPX_STATUS_OK;
#endif
}

class StreamReaderCallbackOwner final : public nvinfer1::IStreamReaderV2
{
public:
    StreamReaderCallbackOwner(const JYPPX_TensorRtLine line, const void* data, const size_t size)
        : line_(line),
          owner_id_(g_stream_reader_owner_next_id.fetch_add(1U, std::memory_order_relaxed)),
          data_(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)
    {
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "stream reader owner created");
    }

    StreamReaderCallbackOwner(const StreamReaderCallbackOwner&) = delete;
    StreamReaderCallbackOwner& operator=(const StreamReaderCallbackOwner&) = delete;
    StreamReaderCallbackOwner(StreamReaderCallbackOwner&&) = delete;
    StreamReaderCallbackOwner& operator=(StreamReaderCallbackOwner&&) = delete;
    ~StreamReaderCallbackOwner() noexcept override = default;

    bool begin_deserialize() noexcept
    {
        std::lock_guard<std::mutex> lock(gate_);
        if (is_deserializing_)
        {
            record_failure_locked(JYPPX_STATUS_INVALID_STATE, "stream reader is already deserializing");
            return false;
        }

        is_deserializing_ = true;
        position_ = 0U;
        ++deserialize_attempt_count_;
        last_status_ = JYPPX_STATUS_OK;
        last_operation_succeeded_ = JYPPX_TRUE;
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "deserialize started at stream position zero");
        return true;
    }

    void end_deserialize(const bool succeeded, const char* diagnostic) noexcept
    {
        std::lock_guard<std::mutex> lock(gate_);
        is_deserializing_ = false;
        last_operation_succeeded_ = succeeded ? JYPPX_TRUE : JYPPX_FALSE;
        last_status_ = succeeded ? JYPPX_STATUS_OK : JYPPX_STATUS_INVALID_STATE;
        if (succeeded)
        {
            ++successful_deserialize_count_;
        }
        else
        {
            ++failed_deserialize_count_;
            ++failure_count_;
        }
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic);
    }

    int64_t read(void* destination, const int64_t nb_bytes, cudaStream_t stream) noexcept override
    {
        CallbackScope callback_scope(*this);
        std::lock_guard<std::mutex> lock(gate_);
        last_had_cuda_stream_ = stream != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        last_read_to_device_ = JYPPX_FALSE;

        if (nb_bytes < 0 || (destination == nullptr && nb_bytes > 0))
        {
            record_failure_locked(JYPPX_STATUS_INVALID_ARGUMENT, "read requires a non-negative size and a destination");
            return -1;
        }

        ++read_count_;
        requested_bytes_ += static_cast<uint64_t>(nb_bytes);
        if (nb_bytes == 0)
        {
            last_status_ = JYPPX_STATUS_OK;
            last_operation_succeeded_ = JYPPX_TRUE;
            copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "zero-length read completed");
            return 0;
        }

        const uint64_t available = static_cast<uint64_t>(data_.size()) - position_;
        const uint64_t requested = static_cast<uint64_t>(nb_bytes);
        const uint64_t copied = std::min(available, requested);
        if (copied == 0U)
        {
            last_status_ = JYPPX_STATUS_OK;
            last_operation_succeeded_ = JYPPX_TRUE;
            copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "end of stream reached");
            return 0;
        }

        bool destination_is_device = false;
        cudaPointerAttributes attributes{};
        const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, destination);
        if (attribute_status == cudaSuccess)
        {
            destination_is_device =
                attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
        }
        else
        {
            (void)cudaGetLastError();
            if (attribute_status != cudaErrorInvalidValue)
            {
                record_failure_locked(JYPPX_STATUS_RUNTIME_ERROR, "cudaPointerGetAttributes failed for stream destination");
                return -1;
            }
        }

        const void* source = data_.data() + static_cast<size_t>(position_);
        if (destination_is_device)
        {
            cudaError_t copy_status = cudaSuccess;
            if (stream != nullptr)
            {
                copy_status = cudaMemcpyAsync(destination, source, static_cast<size_t>(copied), cudaMemcpyHostToDevice, stream);
                if (copy_status == cudaSuccess)
                {
                    copy_status = cudaStreamSynchronize(stream);
                }
            }
            else
            {
                copy_status = cudaMemcpy(destination, source, static_cast<size_t>(copied), cudaMemcpyHostToDevice);
            }

            if (copy_status != cudaSuccess)
            {
                record_failure_locked(JYPPX_STATUS_RUNTIME_ERROR, cudaGetErrorString(copy_status));
                return -1;
            }
            ++device_read_count_;
            last_read_to_device_ = JYPPX_TRUE;
        }
        else
        {
            std::memcpy(destination, source, static_cast<size_t>(copied));
            ++host_read_count_;
        }

        position_ += copied;
        bytes_read_ += copied;
        last_status_ = JYPPX_STATUS_OK;
        last_operation_succeeded_ = JYPPX_TRUE;
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "stream read completed");
        return static_cast<int64_t>(copied);
    }

    bool seek(const int64_t offset, const nvinfer1::SeekPosition where) noexcept override
    {
        CallbackScope callback_scope(*this);
        std::lock_guard<std::mutex> lock(gate_);
        ++seek_count_;
        last_seek_position_ = static_cast<int32_t>(where);

        uint64_t base = 0U;
        switch (where)
        {
        case nvinfer1::SeekPosition::kSET:
            base = 0U;
            break;
        case nvinfer1::SeekPosition::kCUR:
            base = position_;
            break;
        case nvinfer1::SeekPosition::kEND:
            base = static_cast<uint64_t>(data_.size());
            break;
        default:
            record_failure_locked(JYPPX_STATUS_INVALID_ARGUMENT, "seek position is not supported");
            return false;
        }

        uint64_t next = base;
        if (offset >= 0)
        {
            const uint64_t positive_offset = static_cast<uint64_t>(offset);
            if (positive_offset > static_cast<uint64_t>(data_.size()) - std::min(base, static_cast<uint64_t>(data_.size())))
            {
                record_failure_locked(JYPPX_STATUS_INVALID_ARGUMENT, "seek would move beyond the end of the stream");
                return false;
            }
            next = base + positive_offset;
        }
        else
        {
            const uint64_t negative_offset = static_cast<uint64_t>(-(offset + 1)) + 1U;
            if (negative_offset > base)
            {
                record_failure_locked(JYPPX_STATUS_INVALID_ARGUMENT, "seek would move before the beginning of the stream");
                return false;
            }
            next = base - negative_offset;
        }

        if (next > static_cast<uint64_t>(data_.size()))
        {
            record_failure_locked(JYPPX_STATUS_INVALID_ARGUMENT, "seek target is outside the stream");
            return false;
        }

        position_ = next;
        last_status_ = JYPPX_STATUS_OK;
        last_operation_succeeded_ = JYPPX_TRUE;
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), "stream seek completed");
        return true;
    }

    void copy_info(JYPPX_TensorRtStreamReaderOwnerInfo* out_info) const noexcept
    {
        std::lock_guard<std::mutex> lock(gate_);
        std::memset(out_info, 0, sizeof(*out_info));
        out_info->line = static_cast<uint32_t>(line_);
        out_info->owner_id = owner_id_;
        out_info->length = static_cast<uint64_t>(data_.size());
        out_info->position = position_;
        out_info->deserialize_attempt_count = deserialize_attempt_count_;
        out_info->successful_deserialize_count = successful_deserialize_count_;
        out_info->failed_deserialize_count = failed_deserialize_count_;
        out_info->read_count = read_count_;
        out_info->seek_count = seek_count_;
        out_info->host_read_count = host_read_count_;
        out_info->device_read_count = device_read_count_;
        out_info->bytes_read = bytes_read_;
        out_info->requested_bytes = requested_bytes_;
        out_info->failure_count = failure_count_;
        out_info->in_flight_callback_count = in_flight_callback_count_.load(std::memory_order_relaxed);
        out_info->max_in_flight_callback_count = max_in_flight_callback_count_.load(std::memory_order_relaxed);
        out_info->last_status = static_cast<int32_t>(last_status_);
        out_info->last_seek_position = last_seek_position_;
        out_info->last_had_cuda_stream = last_had_cuda_stream_;
        out_info->last_read_to_device = last_read_to_device_;
        out_info->last_operation_succeeded = last_operation_succeeded_;
        out_info->is_deserializing = is_deserializing_ ? JYPPX_TRUE : JYPPX_FALSE;
        copy_stream_reader_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), last_diagnostic_);
    }

private:
    class CallbackScope
    {
    public:
        explicit CallbackScope(StreamReaderCallbackOwner& owner) noexcept : owner_(owner)
        {
            const uint64_t current = owner_.in_flight_callback_count_.fetch_add(1U, std::memory_order_relaxed) + 1U;
            uint64_t observed = owner_.max_in_flight_callback_count_.load(std::memory_order_relaxed);
            while (current > observed &&
                   !owner_.max_in_flight_callback_count_.compare_exchange_weak(
                       observed, current, std::memory_order_relaxed, std::memory_order_relaxed))
            {
            }
        }

        ~CallbackScope() noexcept
        {
            owner_.in_flight_callback_count_.fetch_sub(1U, std::memory_order_relaxed);
        }

    private:
        StreamReaderCallbackOwner& owner_;
    };

    void record_failure_locked(const JYPPX_StatusCode status, const char* diagnostic) noexcept
    {
        ++failure_count_;
        last_status_ = status;
        last_operation_succeeded_ = JYPPX_FALSE;
        copy_stream_reader_text(last_diagnostic_, sizeof(last_diagnostic_), diagnostic);
    }

    const JYPPX_TensorRtLine line_;
    const uint64_t owner_id_;
    const std::vector<uint8_t> data_;
    mutable std::mutex gate_;
    uint64_t position_{0U};
    uint64_t deserialize_attempt_count_{0U};
    uint64_t successful_deserialize_count_{0U};
    uint64_t failed_deserialize_count_{0U};
    uint64_t read_count_{0U};
    uint64_t seek_count_{0U};
    uint64_t host_read_count_{0U};
    uint64_t device_read_count_{0U};
    uint64_t bytes_read_{0U};
    uint64_t requested_bytes_{0U};
    uint64_t failure_count_{0U};
    std::atomic<uint64_t> in_flight_callback_count_{0U};
    std::atomic<uint64_t> max_in_flight_callback_count_{0U};
    JYPPX_StatusCode last_status_{JYPPX_STATUS_OK};
    int32_t last_seek_position_{-1};
    JYPPX_Boolean last_had_cuda_stream_{JYPPX_FALSE};
    JYPPX_Boolean last_read_to_device_{JYPPX_FALSE};
    JYPPX_Boolean last_operation_succeeded_{JYPPX_TRUE};
    bool is_deserializing_{false};
    char last_diagnostic_[1024]{};
};

static_assert(!std::is_copy_constructible<StreamReaderCallbackOwner>::value, "Stream reader owners must not be copyable.");
static_assert(!std::is_move_constructible<StreamReaderCallbackOwner>::value, "Stream reader owners must not be movable.");
static_assert(std::is_nothrow_destructible<StreamReaderCallbackOwner>::value, "Stream reader owner destruction must not throw.");

static JYPPX_StatusCode get_stream_reader_callback_owner(
    JYPPX_TensorRtStreamReaderOwner* owner,
    StreamReaderCallbackOwner** out_payload)
{
    auto status = jyppx::tensorrt::validate_handle(
        owner,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_STREAM_READER_CALLBACK_OWNER,
        "owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<StreamReaderCallbackOwner>(owner);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Stream reader owner handle has no callback payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_STREAM_READER_OWNER_API(stream_reader_v2_owner_create)(
    const void* data,
    const size_t size,
    JYPPX_TensorRtStreamReaderOwner** out_owner)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_owner, "out_owner");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    *out_owner = nullptr;

    if (data == nullptr || size == 0U || size > static_cast<size_t>((std::numeric_limits<int64_t>::max)()))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Stream reader data must be non-empty and fit in int64.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR
    try
    {
        auto* payload = new (std::nothrow) StreamReaderCallbackOwner(kLine, data, size);
        if (payload == nullptr)
        {
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_STREAM_READER_CALLBACK_OWNER,
            payload,
            &destroy_payload<StreamReaderCallbackOwner>);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        *out_owner = reinterpret_cast<JYPPX_TensorRtStreamReaderOwner*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kStreamReaderOwnerFeature, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, kStreamReaderOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kStreamReaderOwnerFeature);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, kStreamReaderOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_STREAM_READER_OWNER_API(stream_reader_v2_owner_get_info)(
    JYPPX_TensorRtStreamReaderOwner* owner,
    JYPPX_TensorRtStreamReaderOwnerInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    std::memset(out_info, 0, sizeof(*out_info));

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR
    StreamReaderCallbackOwner* payload = nullptr;
    status = get_stream_reader_callback_owner(owner, &payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    payload->copy_info(out_info);
    return JYPPX_STATUS_OK;
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kStreamReaderOwnerFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_STREAM_READER_OWNER_API(runtime_deserialize_stream_reader_v2)(
    JYPPX_TensorRtRuntime* runtime,
    JYPPX_TensorRtStreamReaderOwner* owner,
    JYPPX_TensorRtCudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    *out_engine = nullptr;

    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_STREAM_READER_EXPECTED_MAJOR
    StreamReaderCallbackOwner* active_owner = nullptr;
    bool deserialize_started = false;
    try
    {
        auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
        StreamReaderCallbackOwner* owner_payload = nullptr;
        status = get_stream_reader_callback_owner(owner, &owner_payload);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        if (runtime_payload == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime handle has no TensorRT payload.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        if (!owner_payload->begin_deserialize())
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Stream reader owner already has an active deserialize call.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        active_owner = owner_payload;
        deserialize_started = true;

        nvinfer1::ICudaEngine* engine = nullptr;
        status = deserialize_stream_reader_with_seh_guard(runtime_payload, owner_payload, &engine);
        if (status != JYPPX_STATUS_OK)
        {
            owner_payload->end_deserialize(false, "TensorRT raised a native exception while using IStreamReaderV2");
            deserialize_started = false;
            return status;
        }
        if (engine == nullptr)
        {
            owner_payload->end_deserialize(false, "TensorRT returned null while deserializing from IStreamReaderV2");
            deserialize_started = false;
            return report_null_vendor_object("deserializeCudaEngine(IStreamReaderV2)");
        }

        JYPPX_TensorRtObjectBase* handle = nullptr;
        status = create_handle_with_payload(
            &handle,
            JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE,
            engine,
            &destroy_payload<nvinfer1::ICudaEngine>);
        if (status != JYPPX_STATUS_OK)
        {
            owner_payload->end_deserialize(false, "bridge could not retain the engine returned by IStreamReaderV2");
            deserialize_started = false;
            return status;
        }
        owner_payload->end_deserialize(true, "TensorRT deserialized the engine through IStreamReaderV2");
        deserialize_started = false;
        *out_engine = reinterpret_cast<JYPPX_TensorRtCudaEngine*>(handle);
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        if (deserialize_started && active_owner != nullptr)
        {
            active_owner->end_deserialize(false, "native exception interrupted IStreamReaderV2 deserialization");
        }
        return jyppx::tensorrt::report_vendor_exception(kLine, kStreamReaderOwnerFeature, exception.what());
    }
    catch (...)
    {
        if (deserialize_started && active_owner != nullptr)
        {
            active_owner->end_deserialize(false, "unknown native exception interrupted IStreamReaderV2 deserialization");
        }
        return jyppx::tensorrt::report_vendor_exception(kLine, kStreamReaderOwnerFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)owner;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kStreamReaderOwnerFeature);
#else
    (void)owner;
    return jyppx::tensorrt::report_vendor_missing(kLine, kStreamReaderOwnerFeature);
#endif
}
