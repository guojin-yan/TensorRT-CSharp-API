#ifndef JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API must be defined before including execution_context_synchronous_inference.inc"
#endif
#ifndef JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR
#error "JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR must be defined before including execution_context_synchronous_inference.inc"
#endif

namespace
{
constexpr int32_t kMaximumSynchronousBindingCount = 1000000;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR
enum class SynchronousInferenceMode
{
    ExecuteV2,
    LegacyExecute,
    EnqueueV2
};

struct SynchronousInferenceState
{
    nvinfer1::IExecutionContext* context;
    SynchronousInferenceMode mode;
    int32_t batch_size;
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8 && JYPPX_HAS_CUDA_TOOLKIT
    cudaStream_t stream;
#endif
};

static JYPPX_StatusCode collect_synchronous_binding_addresses(
    nvinfer1::IExecutionContext* context,
    SynchronousInferenceMode mode,
    std::vector<void*>* out_bindings)
{
    auto const& engine = context->getEngine();

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    const bool implicit_batch = engine.hasImplicitBatchDimension();
    if (mode == SynchronousInferenceMode::LegacyExecute && !implicit_batch)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Legacy IExecutionContext::execute requires an implicit-batch TensorRT 8 engine; use ExecuteV2 for explicit-batch engines.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    if (mode != SynchronousInferenceMode::LegacyExecute && implicit_batch)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "IExecutionContext::executeV2/enqueueV2 requires an explicit-batch TensorRT 8 engine; use legacy Execute for implicit-batch engines.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t binding_count = engine.getNbBindings();
    const int32_t profile_count = engine.getNbOptimizationProfiles();
    if (binding_count <= 0 || binding_count > kMaximumSynchronousBindingCount || profile_count <= 0 || binding_count % profile_count != 0)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "TensorRT 8 engine binding/profile counts are invalid for synchronous inference.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    int32_t active_profile = context->getOptimizationProfile();
    if (active_profile < 0)
    {
        active_profile = 0;
    }
    if (active_profile >= profile_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context optimization profile is outside the engine profile range.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t bindings_per_profile = binding_count / profile_count;
    const int32_t active_offset = active_profile * bindings_per_profile;
    out_bindings->assign(static_cast<size_t>(binding_count), nullptr);
    for (int32_t local_index = 0; local_index < bindings_per_profile; ++local_index)
    {
        const char* tensor_name = engine.getBindingName(local_index);
        if (tensor_name == nullptr || tensor_name[0] == '\0')
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 returned an empty binding name.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const void* address = context->getTensorAddress(tensor_name);
        if (address == nullptr)
        {
            std::ostringstream builder;
            builder << "Tensor address is not bound for synchronous inference: " << tensor_name;
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            return JYPPX_STATUS_INVALID_STATE;
        }

        (*out_bindings)[static_cast<size_t>(active_offset + local_index)] = const_cast<void*>(address);
    }
#else
    (void)mode;
    const int32_t binding_count = engine.getNbIOTensors();
    if (binding_count <= 0 || binding_count > kMaximumSynchronousBindingCount)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT engine I/O tensor count is invalid for synchronous inference.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    out_bindings->reserve(static_cast<size_t>(binding_count));
    for (int32_t binding_index = 0; binding_index < binding_count; ++binding_index)
    {
        const char* tensor_name = engine.getIOTensorName(binding_index);
        if (tensor_name == nullptr || tensor_name[0] == '\0')
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned an empty I/O tensor name.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const void* address = context->getTensorAddress(tensor_name);
        if (address == nullptr)
        {
            std::ostringstream builder;
            builder << "Tensor address is not bound for synchronous inference: " << tensor_name;
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
            return JYPPX_STATUS_INVALID_STATE;
        }

        out_bindings->push_back(const_cast<void*>(address));
    }
#endif

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode run_synchronous_inference_body(void* raw_state)
{
    auto* state = static_cast<SynchronousInferenceState*>(raw_state);
    std::vector<void*> bindings;
    auto status = collect_synchronous_binding_addresses(state->context, state->mode, &bindings);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    bool succeeded = false;
    switch (state->mode)
    {
    case SynchronousInferenceMode::ExecuteV2:
        succeeded = state->context->executeV2(bindings.data());
        break;
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    case SynchronousInferenceMode::LegacyExecute:
        succeeded = state->context->execute(state->batch_size, bindings.data());
        break;
#if JYPPX_HAS_CUDA_TOOLKIT
    case SynchronousInferenceMode::EnqueueV2:
        succeeded = state->context->enqueueV2(bindings.data(), state->stream, nullptr);
        break;
#endif
#endif
    default:
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unsupported synchronous inference mode.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (!succeeded)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT synchronous/legacy inference call returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode run_synchronous_inference_with_seh_guard(
    SynchronousInferenceState* state,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return run_synchronous_inference_body(state);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return run_synchronous_inference_body(state);
#endif
}

static JYPPX_StatusCode run_synchronous_inference(
    SynchronousInferenceState* state,
    const char* feature_name)
{
    try
    {
        return run_synchronous_inference_with_seh_guard(state, feature_name);
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

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_V2_API(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context executeV2");
    if (status != JYPPX_STATUS_OK) { return status; }
    SynchronousInferenceState state{context_payload, SynchronousInferenceMode::ExecuteV2, 0
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8 && JYPPX_HAS_CUDA_TOOLKIT
        , nullptr
#endif
    };
    return run_synchronous_inference(&state, "execution context executeV2");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context executeV2");
#endif
}

#ifdef JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_LEGACY_API
JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_EXECUTE_LEGACY_API(
    JYPPX_TensorRtExecutionContext* context,
    int32_t batch_size)
{
    if (batch_size <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Legacy execution batch size must be positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR && JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR == 8
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "legacy execution context execute");
    if (status != JYPPX_STATUS_OK) { return status; }
    SynchronousInferenceState state{context_payload, SynchronousInferenceMode::LegacyExecute, batch_size
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8 && JYPPX_HAS_CUDA_TOOLKIT
        , nullptr
#endif
    };
    return run_synchronous_inference(&state, "legacy execution context execute");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "legacy execution context execute");
#endif
}
#endif

#ifdef JYPPX_TRT_EXECUTION_CONTEXT_ENQUEUE_V2_API
JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_ENQUEUE_V2_API(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR && JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR == 8 && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context enqueueV2");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (stream_payload == nullptr || stream_payload->handle == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "CUDA stream handle does not carry a live non-default stream.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    SynchronousInferenceState state{context_payload, SynchronousInferenceMode::EnqueueV2, 0, stream_payload->handle};
    return run_synchronous_inference(&state, "execution context enqueueV2");
#elif JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR && JYPPX_TRT_EXECUTION_CONTEXT_EXPECTED_MAJOR == 8
    (void)context;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context enqueueV2");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueueV2");
#endif
}
#endif
