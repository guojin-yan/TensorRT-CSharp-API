#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
constexpr int32_t kTrt10TimingCacheKeySize = 16;

JYPPX_StatusCode get_trt10_timing_cache_payload(
    JYPPX_TensorRtTimingCache* cache,
    nvinfer1::ITimingCache** out_cache,
    const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache = get_payload<nvinfer1::ITimingCache>(cache);
    if (*out_cache == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode make_trt10_timing_cache_key(const uint8_t* key_data, int32_t key_size, nvinfer1::TimingCacheKey* out_key)
{
    if (key_data == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache key data must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (key_size != kTrt10TimingCacheKeySize)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache keys must be exactly 16 bytes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::memcpy(out_key->data, key_data, kTrt10TimingCacheKeySize);
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt10_timing_cache_combine(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtTimingCache* input_cache, JYPPX_Boolean ignore_mismatch, JYPPX_Boolean* out_combined)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_combined, "out_combined");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_combined = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITimingCache* input_cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(input_cache, &input_cache_payload, "input_cache");
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

JYPPX_StatusCode jyppx_trt10_timing_cache_reset(JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean* out_reset)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reset, "out_reset");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reset = cache_payload->reset() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache reset");
#endif
}

JYPPX_StatusCode jyppx_trt10_timing_cache_query_key_count(JYPPX_TensorRtTimingCache* cache, int64_t* out_key_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_key_count, "out_key_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_key_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int64_t count = cache_payload->queryKeys(nullptr, 0);
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ITimingCache::queryKeys returned an error.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_key_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache key count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_timing_cache_copy_keys(JYPPX_TensorRtTimingCache* cache, uint8_t* output_keys, int64_t key_capacity, int64_t* out_key_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_key_count, "out_key_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_key_count = 0;
    if (key_capacity < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache key capacity must not be negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (key_capacity > 0 && output_keys == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT timing cache output key buffer must not be null when capacity is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<nvinfer1::TimingCacheKey> keys(static_cast<size_t>(key_capacity));
    nvinfer1::TimingCacheKey* key_buffer = keys.empty() ? nullptr : keys.data();
    const int64_t count = cache_payload->queryKeys(key_buffer, key_capacity);
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ITimingCache::queryKeys returned an error.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    const int64_t copy_count = count < key_capacity ? count : key_capacity;
    if (copy_count > 0)
    {
        std::memcpy(output_keys, keys.data(), static_cast<size_t>(copy_count) * kTrt10TimingCacheKeySize);
    }

    *out_key_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)output_keys;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache key copy");
#endif
}

JYPPX_StatusCode jyppx_trt10_timing_cache_query(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t* out_tactic_hash, float* out_timing_msec, JYPPX_Boolean* out_found)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_hash, "out_tactic_hash");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_timing_msec, "out_timing_msec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_found, "out_found");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tactic_hash = 0;
    *out_timing_msec = 0.0F;
    *out_found = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheKey key{};
    status = make_trt10_timing_cache_key(key_data, key_size, &key);
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::TimingCacheValue value = cache_payload->query(key);
    *out_tactic_hash = value.tacticHash;
    *out_timing_msec = value.timingMSec;
    *out_found = value.tacticHash == nvinfer1::TimingCacheValue::kINVALID_TACTIC_HASH ? JYPPX_FALSE : JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)key_data;
    (void)key_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache value query");
#endif
}

JYPPX_StatusCode jyppx_trt10_timing_cache_update(JYPPX_TensorRtTimingCache* cache, const uint8_t* key_data, int32_t key_size, uint64_t tactic_hash, float timing_msec, JYPPX_Boolean* out_updated)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_updated, "out_updated");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_updated = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::ITimingCache* cache_payload = nullptr;
    status = get_trt10_timing_cache_payload(cache, &cache_payload, "cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheKey key{};
    status = make_trt10_timing_cache_key(key_data, key_size, &key);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::TimingCacheValue value{};
    value.tacticHash = tactic_hash;
    value.timingMSec = timing_msec;
    *out_updated = cache_payload->update(key, value) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)cache;
    (void)key_data;
    (void)key_size;
    (void)tactic_hash;
    (void)timing_msec;
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache value update");
#endif
}
