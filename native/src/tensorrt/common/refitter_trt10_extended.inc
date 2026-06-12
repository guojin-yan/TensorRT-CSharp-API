#ifndef JYPPX_TRT_REFITTER_API
#error "JYPPX_TRT_REFITTER_API must be defined before including refitter_trt10_extended.inc"
#endif

#if JYPPX_HAS_TENSORRT
static void copy_refitter_weights_info_for_controls(const nvinfer1::Weights& weights, JYPPX_TensorRtWeightsInfo* out_info)
{
    out_info->data_type = static_cast<int32_t>(weights.type);
    out_info->count = weights.count;
    out_info->has_values = weights.values != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
}
#endif

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_refit_cuda_engine_async)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_CudaStream* stream,
    JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitted = JYPPX_FALSE;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10 && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter async refit CUDA engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (stream_payload == nullptr)
    {
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_refitted = payload->refitCudaEngineAsync(stream_payload->handle) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    (void)refitter;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT refitter async refit CUDA engine");
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter async refit CUDA engine");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter async refit CUDA engine");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_set_weights_validation)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_Boolean enabled)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    auto status = get_refitter_payload_for_controls(refitter, &payload, "refitter weights validation set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    payload->setWeightsValidation(enabled != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    (void)enabled;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter weights validation set");
#else
    (void)refitter;
    (void)enabled;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights validation set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_weights_validation)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter weights validation query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = payload->getWeightsValidation() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter weights validation query");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights validation query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_unset_named_weights)(
    JYPPX_TensorRtRefitter* refitter,
    const char* weights_name,
    JYPPX_Boolean* out_unset)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unset, "out_unset");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_unset = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter named weights unset");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_unset = payload->unsetNamedWeights(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter named weights unset");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter named weights unset");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_weights_location)(
    JYPPX_TensorRtRefitter* refitter,
    const char* weights_name,
    int32_t* out_location)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_location, "out_location");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_location = 0;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter weights location query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_location = static_cast<int32_t>(payload->getWeightsLocation(weights_name));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter weights location query");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights location query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_named_weights_info)(
    JYPPX_TensorRtRefitter* refitter,
    const char* weights_name,
    JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter named weights info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_refitter_weights_info_for_controls(payload->getNamedWeights(weights_name), out_info);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter named weights info query");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter named weights info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_weights_prototype_info)(
    JYPPX_TensorRtRefitter* refitter,
    const char* weights_name,
    JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter weights prototype info query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_refitter_weights_info_for_controls(payload->getWeightsPrototype(weights_name), out_info);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "refitter weights prototype info query");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights prototype info query");
#endif
}

#undef JYPPX_TRT_REFITTER_API
