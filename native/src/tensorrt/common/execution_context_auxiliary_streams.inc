#ifndef JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API must be defined before including execution_context_auxiliary_streams.inc"
#endif

#ifndef JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API must be defined before including execution_context_auxiliary_streams.inc"
#endif

namespace
{
constexpr int32_t kMaximumAuxiliaryStreamCount = 1000000;

#if JYPPX_HAS_TENSORRT
struct AuxiliaryStreamAssignmentState
{
    nvinfer1::IExecutionContext* context;
    cudaStream_t* streams;
    int32_t stream_count;
};

static JYPPX_StatusCode set_auxiliary_streams_body(void* raw_state)
{
    auto* state = static_cast<AuxiliaryStreamAssignmentState*>(raw_state);
    state->context->setAuxStreams(state->streams, state->stream_count);
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode set_auxiliary_streams_with_seh_guard(
    AuxiliaryStreamAssignmentState* state,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return set_auxiliary_streams_body(state);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return set_auxiliary_streams_body(state);
#endif
}

static JYPPX_StatusCode run_auxiliary_stream_assignment(
    AuxiliaryStreamAssignmentState* state,
    const char* feature_name)
{
    try
    {
        return set_auxiliary_streams_with_seh_guard(state, feature_name);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_CudaStream** streams,
    int32_t stream_count)
{
    if (stream_count < 0 || stream_count > kMaximumAuxiliaryStreamCount)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Auxiliary stream count must be in the range [0, 1000000].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (stream_count > 0 && streams == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Auxiliary stream array must be present when stream_count is positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context auxiliary stream assignment");
    if (status != JYPPX_STATUS_OK) { return status; }

    if (stream_count == 0)
    {
        AuxiliaryStreamAssignmentState state{context_payload, nullptr, 0};
        return run_auxiliary_stream_assignment(&state, "execution context auxiliary stream clear");
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    std::vector<cudaStream_t> native_streams;
    native_streams.reserve(static_cast<size_t>(stream_count));
    for (int32_t i = 0; i < stream_count; ++i)
    {
        status = jyppx::cuda::validate_stream(streams[i], "aux_stream");
        if (status != JYPPX_STATUS_OK) { return status; }

        auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(streams[i]);
        if (stream_payload == nullptr || stream_payload->handle == nullptr)
        {
            jyppx::common::set_last_error(
                JYPPX_ERROR_CATEGORY_TENSORRT,
                "Auxiliary streams must be non-default CUDA streams with live bridge handles.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        for (cudaStream_t existing : native_streams)
        {
            if (existing == stream_payload->handle)
            {
                jyppx::common::set_last_error(
                    JYPPX_ERROR_CATEGORY_TENSORRT,
                    "Auxiliary streams must be unique to avoid TensorRT cross-stream deadlocks.");
                return JYPPX_STATUS_INVALID_ARGUMENT;
            }
        }
        native_streams.push_back(stream_payload->handle);
    }

    AuxiliaryStreamAssignmentState state{context_payload, native_streams.data(), stream_count};
    return run_auxiliary_stream_assignment(&state, "execution context auxiliary stream assignment");
#else
    (void)streams;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context auxiliary stream assignment");
#endif
#else
    (void)context;
    (void)streams;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context auxiliary stream assignment");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API(
    JYPPX_TensorRtExecutionContext* context)
{
    return JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API(context, nullptr, 0);
}
