JYPPX_StatusCode jyppx_trt8_builder_config_add_optimization_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile, int32_t* out_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_index, "out_index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_index = -1;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config add optimization profile");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (config_payload == nullptr || profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t index = config_payload->addOptimizationProfile(profile_payload);
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::addOptimizationProfile returned a negative index.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    *out_index = index;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config add optimization profile");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_profile_stream(JYPPX_TensorRtBuilderConfig* config, JYPPX_CudaStream* stream)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config profile stream set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (config_payload == nullptr || stream_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or CUDA stream handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setProfileStream(stream_payload->handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profile stream set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_is_profile_stream_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config profile stream query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_set = config_payload->getProfileStream() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config profile stream query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_optimization_profile_count(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config optimization profile count query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = config_payload->getNbOptimizationProfiles();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config optimization profile count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtOptimizationProfile* profile)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config calibration profile set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (config_payload == nullptr || profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!config_payload->setCalibrationProfile(profile_payload))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::setCalibrationProfile returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config calibration profile set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_has_calibration_profile(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_profile)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_profile, "out_has_profile");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_profile = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config calibration profile query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_has_profile = config_payload->getCalibrationProfile() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config calibration profile query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_has_algorithm_selector(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_selector)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_selector, "out_has_selector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_selector = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config algorithm selector presence query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_has_selector = config_payload->getAlgorithmSelector() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config algorithm selector presence query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_has_int8_calibrator(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_calibrator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_calibrator, "out_has_calibrator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_calibrator = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config INT8 calibrator presence query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_has_calibrator = config_payload->getInt8Calibrator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config INT8 calibrator presence query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean enabled)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config flag set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const auto builder_flag = static_cast<nvinfer1::BuilderFlag>(flag);
    if (enabled == JYPPX_FALSE)
    {
        config_payload->clearFlag(builder_flag);
    }
    else
    {
        config_payload->setFlag(builder_flag);
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flag set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config flag query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = config_payload->getFlag(static_cast<nvinfer1::BuilderFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flag query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config flags query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_flags = static_cast<uint32_t>(config_payload->getFlags());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flags query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t capability)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (capability < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine capability must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config engine capability set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setEngineCapability(static_cast<nvinfer1::EngineCapability>(capability));
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config engine capability set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_engine_capability(JYPPX_TensorRtBuilderConfig* config, int32_t* out_capability)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_capability, "out_capability");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capability = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config engine capability query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_capability = static_cast<int32_t>(config_payload->getEngineCapability());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config engine capability query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean enabled)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (feature < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Preview feature must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config preview feature set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setPreviewFeature(static_cast<nvinfer1::PreviewFeature>(feature), enabled != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)feature;
    (void)enabled;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config preview feature set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_preview_feature(JYPPX_TensorRtBuilderConfig* config, int32_t feature, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (feature < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Preview feature must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config preview feature query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = config_payload->getPreviewFeature(static_cast<nvinfer1::PreviewFeature>(feature)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config preview feature query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t level)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (level < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Hardware compatibility level must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config hardware compatibility level set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setHardwareCompatibilityLevel(static_cast<nvinfer1::HardwareCompatibilityLevel>(level));
    return JYPPX_STATUS_OK;
#else
    (void)level;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config hardware compatibility level set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_hardware_compatibility_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_level = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config hardware compatibility level query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_level = static_cast<int32_t>(config_payload->getHardwareCompatibilityLevel());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config hardware compatibility level query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t platform)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    (void)platform;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime platform is available only on TensorRT 10.x.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_runtime_platform(JYPPX_TensorRtBuilderConfig* config, int32_t* out_platform)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_platform, "out_platform");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_platform = 0;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime platform is available only on TensorRT 10.x.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t pool_size)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (pool < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Memory pool type must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config memory pool limit set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool), pool_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config memory pool limit set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_memory_pool_limit(JYPPX_TensorRtBuilderConfig* config, int32_t pool, size_t* out_pool_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_pool_size, "out_pool_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (pool < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Memory pool type must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_pool_size = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config memory pool limit query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_pool_size = config_payload->getMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config memory pool limit query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (level < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder optimization level must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder optimization level set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setBuilderOptimizationLevel(level);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder optimization level set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_level = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder optimization level query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_level = config_payload->getBuilderOptimizationLevel();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder optimization level query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t verbosity)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (verbosity < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profiling verbosity must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "profiling verbosity set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setProfilingVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "profiling verbosity set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_profiling_verbosity(JYPPX_TensorRtBuilderConfig* config, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_verbosity = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "profiling verbosity query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_verbosity = static_cast<int32_t>(config_payload->getProfilingVerbosity());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "profiling verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t stream_count)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (stream_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Max auxiliary stream count must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "max aux streams set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setMaxAuxStreams(stream_count);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "max aux streams set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_max_aux_streams(JYPPX_TensorRtBuilderConfig* config, int32_t* out_stream_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_stream_count, "out_stream_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream_count = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "max aux streams query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_stream_count = config_payload->getMaxAuxStreams();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "max aux streams query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t iterations)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (iterations <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Average timing iterations must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "average timing iterations set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setAvgTimingIterations(iterations);
    return JYPPX_STATUS_OK;
#else
    (void)iterations;
    return jyppx::tensorrt::report_vendor_missing(kLine, "average timing iterations set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_average_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t* out_iterations)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_iterations, "out_iterations");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_iterations = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "average timing iterations query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_iterations = config_payload->getAvgTimingIterations();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "average timing iterations query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_max_workspace_size(JYPPX_TensorRtBuilderConfig* config, size_t* out_workspace_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_workspace_size, "out_workspace_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_workspace_size = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_workspace_size = config_payload->getMaxWorkspaceSize();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config legacy max workspace size query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config legacy max workspace size query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_min_timing_iterations(JYPPX_TensorRtBuilderConfig* config, int32_t* out_iterations)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_iterations, "out_iterations");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_iterations = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_iterations = config_payload->getMinTimingIterations();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config legacy min timing iterations query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config legacy min timing iterations query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t tactic_sources)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "tactic sources set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!config_payload->setTacticSources(tactic_sources))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT rejected the tactic source bitmask.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
#else
    (void)tactic_sources;
    return jyppx::tensorrt::report_vendor_missing(kLine, "tactic sources set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_tactic_sources(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_tactic_sources)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tactic_sources, "out_tactic_sources");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tactic_sources = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "tactic sources query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_tactic_sources = config_payload->getTacticSources();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "tactic sources query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_nb_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = config_payload->getNbPluginsToSerialize();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config plugin serialization count query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config plugin serialization count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t device_type)
{
    if (device_type < 0 || device_type > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT device type must be 0 (GPU) or 1 (DLA).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config default device type set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(device_type));
    return JYPPX_STATUS_OK;
#else
    (void)device_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config default device type set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t* out_device_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_device_type, "out_device_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device_type = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config default device type query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_device_type = static_cast<int32_t>(config_payload->getDefaultDeviceType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config default device type query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t dla_core)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config DLA core set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setDLACore(dla_core);
    return JYPPX_STATUS_OK;
#else
    (void)dla_core;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config DLA core set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t* out_dla_core)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dla_core, "out_dla_core");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_dla_core = -1;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config DLA core query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_dla_core = config_payload->getDLACore();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config DLA core query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_can_run_on_dla(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_can_run)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_can_run, "out_can_run");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_can_run = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "layer DLA capability query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* layer_payload = get_layer_payload(layer);
    if (config_payload == nullptr || layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_can_run = config_payload->canRunOnDLA(layer_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer DLA capability query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t device_type)
{
    if (device_type < 0 || device_type > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT device type must be 0 (GPU) or 1 (DLA).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "layer device type set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* layer_payload = get_layer_payload(layer);
    if (config_payload == nullptr || layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setDeviceType(layer_payload, static_cast<nvinfer1::DeviceType>(device_type));
    return JYPPX_STATUS_OK;
#else
    (void)device_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, int32_t* out_device_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_device_type, "out_device_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device_type = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "layer device type query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* layer_payload = get_layer_payload(layer);
    if (config_payload == nullptr || layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_device_type = static_cast<int32_t>(config_payload->getDeviceType(layer_payload));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_is_layer_device_type_set(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_is_set = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "layer device type set-state query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* layer_payload = get_layer_payload(layer);
    if (config_payload == nullptr || layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_is_set = config_payload->isDeviceTypeSet(layer_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type set-state query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_reset_layer_device_type(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "layer device type reset");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* layer_payload = get_layer_payload(layer);
    if (config_payload == nullptr || layer_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or layer handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->resetDeviceType(layer_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer device type reset");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_create_timing_cache(JYPPX_TensorRtBuilderConfig* config, const void* blob, size_t blob_size, JYPPX_TensorRtTimingCache** out_cache)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cache, "out_cache");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (blob == nullptr && blob_size != 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Timing cache blob must not be null when blob_size is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_cache = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "timing cache creation");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ITimingCache* cache = config_payload->createTimingCache(blob, blob_size);
    if (cache == nullptr)
    {
        return report_null_vendor_object("createTimingCache");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, cache, &destroy_payload<nvinfer1::ITimingCache>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cache = reinterpret_cast<JYPPX_TensorRtTimingCache*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache creation");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_timing_cache(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtTimingCache* cache, JYPPX_Boolean ignore_mismatch)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, "cache");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "timing cache attach");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* cache_payload = get_payload<nvinfer1::ITimingCache>(cache);
    if (config_payload == nullptr || cache_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or timing cache handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (!config_payload->setTimingCache(*cache_payload, ignore_mismatch != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBuilderConfig::setTimingCache returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache attach");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_quantization_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t flags)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config quantization flags set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setQuantizationFlags(static_cast<nvinfer1::QuantizationFlags>(flags));
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config quantization flags set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_quantization_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config quantization flags query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_flags = static_cast<uint32_t>(config_payload->getQuantizationFlags());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config quantization flags query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_clear_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Quantization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config quantization flag clear");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->clearQuantizationFlag(static_cast<nvinfer1::QuantizationFlag>(flag));
    return JYPPX_STATUS_OK;
#else
    (void)flag;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config quantization flag clear");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Quantization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config quantization flag set");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    config_payload->setQuantizationFlag(static_cast<nvinfer1::QuantizationFlag>(flag));
    return JYPPX_STATUS_OK;
#else
    (void)flag;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config quantization flag set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_quantization_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Quantization flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config quantization flag query");
    }

    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = config_payload->getQuantizationFlag(static_cast<nvinfer1::QuantizationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)flag;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config quantization flag query");
#endif
}

JYPPX_StatusCode jyppx_trt8_timing_cache_serialize(JYPPX_TensorRtTimingCache* cache, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(cache, kLine, JYPPX_TENSORRT_OBJECT_KIND_TIMING_CACHE, "cache");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 8)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "timing cache serialization");
    }

    auto* cache_payload = get_payload<nvinfer1::ITimingCache>(cache);
    if (cache_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Timing cache handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IHostMemory* host_memory = cache_payload->serialize();
    if (host_memory == nullptr)
    {
        return report_null_vendor_object("ITimingCache::serialize");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, host_memory, &destroy_host_memory_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_host_memory = reinterpret_cast<JYPPX_TensorRtHostMemory*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "timing cache serialization");
#endif
}
