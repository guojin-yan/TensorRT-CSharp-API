#ifndef JYPPX_TRT_ERROR_CODE_METADATA_API
#error "JYPPX_TRT_ERROR_CODE_METADATA_API must be defined before including error_code_metadata.inc"
#endif

JYPPX_StatusCode JYPPX_TRT_ERROR_CODE_METADATA_API(int32_t* out_exclusive_upper_bound)
{
    auto status = jyppx::tensorrt::validate_output_pointer(
        out_exclusive_upper_bound,
        "out_exclusive_upper_bound");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_exclusive_upper_bound = 0;

#if JYPPX_HAS_TENSORRT
    *out_exclusive_upper_bound = nvinfer1::EnumMax<nvinfer1::ErrorCode>();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "TensorRT error-code enum metadata query");
#endif
}
