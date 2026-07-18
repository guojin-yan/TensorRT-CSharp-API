#ifndef JYPPX_TRT_SAFE_PLUGIN_PREFIX
#error "JYPPX_TRT_SAFE_PLUGIN_PREFIX must be defined before including safe_deferred_plugin_initialization.inc"
#endif

#define JYPPX_TRT_SAFE_PLUGIN_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_SAFE_PLUGIN_CONCAT(prefix, name) JYPPX_TRT_SAFE_PLUGIN_CONCAT2(prefix, name)
#define JYPPX_TRT_SAFE_PLUGIN_FN(name) JYPPX_TRT_SAFE_PLUGIN_CONCAT(JYPPX_TRT_SAFE_PLUGIN_PREFIX, name)

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR
static JYPPX_StatusCode JYPPX_TRT_SAFE_PLUGIN_FN(init_with_seh_guard)(
    ManagedLogger* logger,
    const char* lib_namespace,
    JYPPX_Boolean* out_initialized)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_initialized = initLibNvInferPlugins(static_cast<void*>(logger), lib_namespace) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_initialized = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, "TensorRT plugin library initialization", seh_exception_code);
    }
#else
    *out_initialized = initLibNvInferPlugins(static_cast<void*>(logger), lib_namespace) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_SAFE_PLUGIN_FN(global_init_lib_nvinfer_plugins)(
    JYPPX_TensorRtLogger* logger,
    const char* lib_namespace,
    JYPPX_Boolean* out_initialized)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_initialized, "out_initialized");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_initialized = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == JYPPX_TRT_SAFE_PLUGIN_EXPECTED_MAJOR
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger handle does not carry a TensorRT logger payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return JYPPX_TRT_SAFE_PLUGIN_FN(init_with_seh_guard)(logger_payload, lib_namespace, out_initialized);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "TensorRT plugin library initialization", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "TensorRT plugin library initialization", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "TensorRT plugin library initialization");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "TensorRT plugin library initialization");
#endif
}

#undef JYPPX_TRT_SAFE_PLUGIN_FN
#undef JYPPX_TRT_SAFE_PLUGIN_CONCAT
#undef JYPPX_TRT_SAFE_PLUGIN_CONCAT2
