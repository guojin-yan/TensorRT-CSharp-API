JYPPX_StatusCode jyppx_trt8_global_get_infer_lib_version(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    *out_version = ::getInferLibVersion();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "TensorRT infer library packed version query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "TensorRT infer library packed version query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_get_onnx_parser_version(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
#if JYPPX_HAS_TENSORRT_ONNXPARSER
    *out_version = ::getNvOnnxParserVersion();
    return JYPPX_STATUS_OK;
#else
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser vendor API is not available.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "TensorRT ONNX parser version query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "TensorRT ONNX parser version query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_has_logger(JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_logger = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    *out_has_logger = ::getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "TensorRT global logger snapshot query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "TensorRT global logger snapshot query");
#endif
}
