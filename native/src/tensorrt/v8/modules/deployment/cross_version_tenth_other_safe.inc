JYPPX_StatusCode jyppx_trt8_host_memory_get_type(JYPPX_TensorRtHostMemory* host_memory, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = -1;
    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "host memory data type query");
#endif

    auto* payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Host memory handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_data_type = static_cast<int32_t>(payload->type());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory data type query");
#endif
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode get_trt8_timing_cache_payload(
    JYPPX_TensorRtTimingCache* cache,
    nvinfer1::ITimingCache** out_cache,
    const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache = get_payload<nvinfer1::ITimingCache>(cache);
    if (*out_cache == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Timing cache handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt8_timing_cache_combine(
    JYPPX_TensorRtTimingCache* cache,
    JYPPX_TensorRtTimingCache* input_cache,
    JYPPX_Boolean ignore_mismatch,
    JYPPX_Boolean* out_combined)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_combined, "out_combined");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_combined = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "timing cache combine");
#endif

    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt8_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITimingCache* input_cache_payload = nullptr;
    status = get_trt8_timing_cache_payload(input_cache, &input_cache_payload, "input_cache");
    if (status != JYPPX_STATUS_OK) { return status; }

    *out_combined = cache_payload->combine(*input_cache_payload, ignore_mismatch != JYPPX_FALSE) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)input_cache;
    (void)ignore_mismatch;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache combine");
#endif
}

JYPPX_StatusCode jyppx_trt8_timing_cache_reset(
    JYPPX_TensorRtTimingCache* cache,
    JYPPX_Boolean* out_reset)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reset, "out_reset");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "timing cache reset");
#endif

    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt8_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = cache_payload->reset() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache reset");
#endif
}
