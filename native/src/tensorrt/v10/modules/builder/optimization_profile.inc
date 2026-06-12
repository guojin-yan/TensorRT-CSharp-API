JYPPX_StatusCode jyppx_trt10_optimization_profile_set_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const int32_t* values, int32_t value_count)
{
    auto status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile shape values must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape values set");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = profile_payload->setShapeValues(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), values, value_count);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setShapeValues returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values set");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_shape_value_count(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = -1;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape value count query");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = profile_payload->getNbShapeValues(input_name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape value count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_shape_values(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int32_t* output_values, int32_t output_count, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (output_count < 0 || (output_values == nullptr && output_count > 0))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile shape value output buffer is invalid.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_count = -1;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape values query");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t required_count = profile_payload->getNbShapeValues(input_name);
    *out_count = required_count;
    if (required_count <= 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_count < required_count || output_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile shape value output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    const int32_t* native_values = profile_payload->getShapeValues(input_name, static_cast<nvinfer1::OptProfileSelector>(selector));
    if (native_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::getShapeValues returned null.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    std::memcpy(output_values, native_values, static_cast<size_t>(required_count) * sizeof(int32_t));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values query");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_set_shape_values_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, const int64_t* values, int32_t value_count, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile V2 shape values must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = profile_payload->setShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), values, value_count);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setShapeValuesV2 returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_set = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape values V2 set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values V2 set");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_shape_value_count_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t* out_count)
{
    return jyppx_trt10_optimization_profile_get_shape_value_count(profile, input_name, out_count);
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_shape_values_v2(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int64_t* output_values, int32_t output_count, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = -1;

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be 0 (min), 1 (opt), or 2 (max).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (output_count < 0 || (output_values == nullptr && output_count > 0))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile V2 shape value output buffer is invalid.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t required_count = profile_payload->getNbShapeValues(input_name);
    *out_count = required_count;
    if (required_count <= 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_count < required_count || output_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile V2 shape value output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    const int64_t* native_values = profile_payload->getShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector));
    if (native_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::getShapeValuesV2 returned null.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    std::memcpy(output_values, native_values, static_cast<size_t>(required_count) * sizeof(int64_t));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile shape values V2 query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values V2 query");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_set_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float target)
{
    auto status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile extra memory target set");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = profile_payload->setExtraMemoryTarget(target);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setExtraMemoryTarget returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile extra memory target set");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_get_extra_memory_target(JYPPX_TensorRtOptimizationProfile* profile, float* out_target)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_target, "out_target");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_target = 0.0F;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile extra memory target query");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_target = profile_payload->getExtraMemoryTarget();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile extra memory target query");
#endif
}

JYPPX_StatusCode jyppx_trt10_optimization_profile_is_valid(JYPPX_TensorRtOptimizationProfile* profile, JYPPX_Boolean* out_is_valid)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_valid, "out_is_valid");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_valid = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "optimization profile validity query");
    }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_valid = profile_payload->isValid() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile validity query");
#endif
}
