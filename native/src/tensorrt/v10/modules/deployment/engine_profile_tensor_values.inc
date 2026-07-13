JYPPX_StatusCode jyppx_trt10_engine_get_profile_tensor_values(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    int32_t selector,
    int32_t* output_values,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (profile_index < 0 || selector < 0 || selector > 2 || output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values require a non-negative profile index, selector in [0,2], and a non-negative output count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile tensor values query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t* values = engine_payload->getProfileTensorValues(
        tensor_name,
        profile_index,
        static_cast<nvinfer1::OptProfileSelector>(selector));
    if (values == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_count == 0 || output_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values output buffer is required when TensorRT returns values.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    for (int32_t i = 0; i < output_count; ++i)
    {
        output_values[i] = values[i];
    }

    *out_count = output_count;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)engine;
    (void)output_values;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine profile tensor values query");
#else
    (void)engine;
    (void)profile_index;
    (void)selector;
    (void)output_values;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile tensor values query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_get_profile_tensor_values_v2(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    int32_t selector,
    int64_t* output_values,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (profile_index < 0 || selector < 0 || selector > 2 || output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values V2 require a non-negative profile index, selector in [0,2], and a non-negative output count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile tensor values V2 query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int64_t* values = engine_payload->getProfileTensorValuesV2(
        tensor_name,
        profile_index,
        static_cast<nvinfer1::OptProfileSelector>(selector));
    if (values == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_count == 0 || output_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values V2 output buffer is required when TensorRT returns values.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    for (int32_t i = 0; i < output_count; ++i)
    {
        output_values[i] = values[i];
    }

    *out_count = output_count;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)engine;
    (void)output_values;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine profile tensor values V2 query");
#else
    (void)engine;
    (void)profile_index;
    (void)selector;
    (void)output_values;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile tensor values V2 query");
#endif
}
