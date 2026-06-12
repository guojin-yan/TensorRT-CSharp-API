JYPPX_StatusCode jyppx_trt10_host_memory_get_type(JYPPX_TensorRtHostMemory* host_memory, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = -1;
    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_data_type = static_cast<int32_t>(payload->type());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory data type query");
#endif
}
