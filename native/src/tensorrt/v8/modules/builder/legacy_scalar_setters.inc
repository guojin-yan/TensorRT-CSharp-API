#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
static JYPPX_StatusCode get_trt8_builder_for_legacy_setter(
    JYPPX_TensorRtBuilder* builder,
    nvinfer1::IBuilder** out_builder)
{
    auto status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_builder = get_payload<nvinfer1::IBuilder>(builder);
    if (*out_builder == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_config_for_legacy_setter(
    JYPPX_TensorRtBuilderConfig* config,
    nvinfer1::IBuilderConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_config = get_payload<nvinfer1::IBuilderConfig>(config);
    if (*out_config == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode set_trt8_builder_max_batch_size_with_seh_guard(
    nvinfer1::IBuilder* builder,
    const int32_t max_batch_size)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try { builder->setMaxBatchSize(max_batch_size); return JYPPX_STATUS_OK; }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    { return jyppx::tensorrt::report_vendor_seh_exception(kLine, "builder legacy max batch size set", seh_exception_code); }
#else
    builder->setMaxBatchSize(max_batch_size);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_trt8_builder_config_flags_with_seh_guard(
    nvinfer1::IBuilderConfig* config,
    const uint32_t flags)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try { config->setFlags(flags); return JYPPX_STATUS_OK; }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    { return jyppx::tensorrt::report_vendor_seh_exception(kLine, "builder config flags set", seh_exception_code); }
#else
    config->setFlags(flags);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_trt8_builder_config_workspace_with_seh_guard(
    nvinfer1::IBuilderConfig* config,
    const size_t workspace_size)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try { config->setMaxWorkspaceSize(workspace_size); return JYPPX_STATUS_OK; }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    { return jyppx::tensorrt::report_vendor_seh_exception(kLine, "builder config legacy max workspace size set", seh_exception_code); }
#else
    config->setMaxWorkspaceSize(workspace_size);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_trt8_builder_config_min_timing_with_seh_guard(
    nvinfer1::IBuilderConfig* config,
    const int32_t iterations)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try { config->setMinTimingIterations(iterations); return JYPPX_STATUS_OK; }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    { return jyppx::tensorrt::report_vendor_seh_exception(kLine, "builder config legacy minimum timing iterations set", seh_exception_code); }
#else
    config->setMinTimingIterations(iterations);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode jyppx_trt8_builder_set_max_batch_size(JYPPX_TensorRtBuilder* builder, const int32_t max_batch_size)
{
    if (max_batch_size < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder max batch size must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IBuilder* payload = nullptr;
    auto status = get_trt8_builder_for_legacy_setter(builder, &payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return set_trt8_builder_max_batch_size_with_seh_guard(payload, max_batch_size);
    }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder legacy max batch size set", exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder legacy max batch size set", "unknown native exception"); }
#elif JYPPX_HAS_TENSORRT
    (void)builder;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder legacy max batch size set");
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder legacy max batch size set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_flags(JYPPX_TensorRtBuilderConfig* config, const uint32_t flags)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IBuilderConfig* payload = nullptr;
    auto status = get_trt8_config_for_legacy_setter(config, &payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return set_trt8_builder_config_flags_with_seh_guard(payload, flags);
    }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config flags set", exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config flags set", "unknown native exception"); }
#elif JYPPX_HAS_TENSORRT
    (void)config; (void)flags;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config flags set");
#else
    (void)config; (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flags set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_max_workspace_size(JYPPX_TensorRtBuilderConfig* config, const size_t workspace_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IBuilderConfig* payload = nullptr;
    auto status = get_trt8_config_for_legacy_setter(config, &payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return set_trt8_builder_config_workspace_with_seh_guard(payload, workspace_size);
    }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config legacy max workspace size set", exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config legacy max workspace size set", "unknown native exception"); }
#elif JYPPX_HAS_TENSORRT
    (void)config; (void)workspace_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config legacy max workspace size set");
#else
    (void)config; (void)workspace_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config legacy max workspace size set");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_min_timing_iterations(JYPPX_TensorRtBuilderConfig* config, const int32_t iterations)
{
    if (iterations < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config minimum timing iterations must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IBuilderConfig* payload = nullptr;
    auto status = get_trt8_config_for_legacy_setter(config, &payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return set_trt8_builder_config_min_timing_with_seh_guard(payload, iterations);
    }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config legacy minimum timing iterations set", exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, "builder config legacy minimum timing iterations set", "unknown native exception"); }
#elif JYPPX_HAS_TENSORRT
    (void)config;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "builder config legacy minimum timing iterations set");
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config legacy minimum timing iterations set");
#endif
}
