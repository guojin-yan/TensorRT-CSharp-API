#ifndef JYPPX_TRT_DIRECT_ENGINE_BUILD_API
#error "JYPPX_TRT_DIRECT_ENGINE_BUILD_API must be defined before including direct_engine_build.inc"
#endif

#ifndef JYPPX_TRT_DIRECT_ENGINE_DESTROY
#error "JYPPX_TRT_DIRECT_ENGINE_DESTROY must be defined before including direct_engine_build.inc"
#endif

namespace
{
#if JYPPX_HAS_TENSORRT
struct DirectEngineBuildState
{
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::ICudaEngine* engine;
};

static JYPPX_StatusCode direct_engine_build_body(void* raw_state)
{
    auto* state = static_cast<DirectEngineBuildState*>(raw_state);
    auto status = ensure_minimal_identity_network(*state->network);
    if (status != JYPPX_STATUS_OK) { return status; }
    state->engine = state->builder->buildEngineWithConfig(*state->network, *state->config);
    return state->engine == nullptr ? JYPPX_STATUS_RUNTIME_ERROR : JYPPX_STATUS_OK;
}

static JYPPX_StatusCode direct_engine_build_with_seh_guard(
    DirectEngineBuildState* state,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return direct_engine_build_body(state);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        state->engine = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return direct_engine_build_body(state);
#endif
}
static JYPPX_StatusCode run_direct_engine_build(
    DirectEngineBuildState* state,
    const char* feature_name)
{
    try
    {
        return direct_engine_build_with_seh_guard(state, feature_name);
    }
    catch (const std::exception& exception)
    {
        state->engine = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        state->engine = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_DIRECT_ENGINE_BUILD_API(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_TensorRtCudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_engine = nullptr;

    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (builder_payload == nullptr || network_payload == nullptr || config_payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "One or more direct engine build handles do not carry the expected TensorRT payloads.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    DirectEngineBuildState state{builder_payload, network_payload, config_payload, nullptr};
    status = run_direct_engine_build(&state, "builder direct engine build");
    if (status != JYPPX_STATUS_OK)
    {
        return status == JYPPX_STATUS_RUNTIME_ERROR
            ? report_null_vendor_object("buildEngineWithConfig")
            : status;
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(
        &handle,
        JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE,
        state.engine,
        &JYPPX_TRT_DIRECT_ENGINE_DESTROY);
    if (status != JYPPX_STATUS_OK)
    {
        JYPPX_TRT_DIRECT_ENGINE_DESTROY(state.engine);
        return status;
    }

    *out_engine = reinterpret_cast<JYPPX_TensorRtCudaEngine*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)network;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder direct engine build");
#endif
}
