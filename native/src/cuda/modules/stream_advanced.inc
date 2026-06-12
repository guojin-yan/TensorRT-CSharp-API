JYPPX_StatusCode jyppx_cuda_stream_attach_mem_async(JYPPX_CudaStream* stream, void* pointer, size_t length, uint32_t flags)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (pointer == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamAttachMemAsync", 0, "invalid-pointer", "CUDA stream memory attachment pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaStreamAttachMemAsync(stream_object->handle, pointer, length, flags), "cudaStreamAttachMemAsync");
#else
    (void)length;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream memory attachment");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_attribute(JYPPX_CudaStream* stream, int32_t attribute, void* value)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (value == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamGetAttribute", 0, "invalid-value", "CUDA stream attribute output value must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaStreamGetAttribute(stream_object->handle, static_cast<cudaStreamAttrID>(attribute), reinterpret_cast<cudaStreamAttrValue*>(value)),
        "cudaStreamGetAttribute");
#else
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_set_attribute(JYPPX_CudaStream* stream, int32_t attribute, const void* value)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (value == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamSetAttribute", 0, "invalid-value", "CUDA stream attribute value must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaStreamSetAttribute(stream_object->handle, static_cast<cudaStreamAttrID>(attribute), reinterpret_cast<const cudaStreamAttrValue*>(value)),
        "cudaStreamSetAttribute");
#else
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream attribute set");
#endif
}
