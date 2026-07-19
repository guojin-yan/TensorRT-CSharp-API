namespace
{
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
void reset_dev_resource_snapshot(JYPPX_CudaDevResourceSnapshot* snapshot)
{
    *snapshot = JYPPX_CudaDevResourceSnapshot{};
    snapshot->type = 0;
    snapshot->device_ordinal = -1;
}

JYPPX_StatusCode copy_dev_resource_snapshot(
    const cudaDevResource& resource,
    JYPPX_CudaDevResourceSnapshot* snapshot)
{
    reset_dev_resource_snapshot(snapshot);
    snapshot->type = static_cast<int32_t>(resource.type);
    snapshot->is_valid = resource.type != cudaDevResourceTypeInvalid ? JYPPX_TRUE : JYPPX_FALSE;
    snapshot->has_next_resource = resource.nextResource != nullptr ? JYPPX_TRUE : JYPPX_FALSE;

    switch (resource.type)
    {
    case cudaDevResourceTypeSm:
        snapshot->sm_count = resource.sm.smCount;
        snapshot->min_sm_partition_size = resource.sm.minSmPartitionSize;
        snapshot->sm_coscheduled_alignment = resource.sm.smCoscheduledAlignment;
        snapshot->sm_flags = resource.sm.flags;
        break;
    case cudaDevResourceTypeWorkqueueConfig:
        snapshot->device_ordinal = resource.wqConfig.device;
        snapshot->workqueue_concurrency_limit = resource.wqConfig.wqConcurrencyLimit;
        snapshot->workqueue_sharing_scope = static_cast<int32_t>(resource.wqConfig.sharingScope);
        break;
    case cudaDevResourceTypeWorkqueue:
        snapshot->has_opaque_workqueue = JYPPX_TRUE;
        break;
    case cudaDevResourceTypeInvalid:
        break;
    default:
        jyppx::cuda::set_cuda_error(
            "cudaDevResourceSnapshot",
            0,
            "unknown-resource-type",
            "CUDA returned a device resource type that is not represented by the pointer-free snapshot.");
        return JYPPX_STATUS_NOT_SUPPORTED;
    }

    return JYPPX_STATUS_OK;
}

template <typename TCallable>
JYPPX_StatusCode query_dev_resource_snapshot(
    const char* operation,
    JYPPX_CudaDevResourceSnapshot* out_snapshot,
    TCallable&& callable)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    reset_dev_resource_snapshot(out_snapshot);

    try
    {
        cudaDevResource resource{};
        status = jyppx::cuda::map_cuda_status(callable(resource), operation);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
        return copy_dev_resource_snapshot(resource, out_snapshot);
    }
    catch (const std::bad_alloc&)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "native-allocation-failed", "The CUDA device resource snapshot bridge could not allocate temporary storage.");
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }
    catch (...)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "native-exception-caught", "A native exception was contained by the CUDA device resource snapshot bridge.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
}
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_dev_resource_snapshot_safe(
    const int32_t device,
    const int32_t resource_type,
    JYPPX_CudaDevResourceSnapshot* out_snapshot)
{
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return query_dev_resource_snapshot("cudaDeviceGetDevResource", out_snapshot, [&](cudaDevResource& resource) {
        return cudaDeviceGetDevResource(device, &resource, static_cast<cudaDevResourceType>(resource_type));
    });
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)device;
    (void)resource_type;
    if (out_snapshot != nullptr) *out_snapshot = JYPPX_CudaDevResourceSnapshot{};
    jyppx::cuda::set_cuda_error("cudaDeviceGetDevResource", 0, "cuda-version-not-supported", "CUDA device resource queries require CUDA Toolkit 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)device;
    (void)resource_type;
    (void)out_snapshot;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device resource snapshot query");
#endif
}

JYPPX_StatusCode jyppx_cuda_execution_context_get_dev_resource_snapshot_safe(
    JYPPX_CudaExecutionContext* context_handle,
    const int32_t resource_type,
    JYPPX_CudaDevResourceSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_execution_context(context_handle, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    auto* context = reinterpret_cast<ExecutionContextObject*>(context_handle);
    return query_dev_resource_snapshot("cudaExecutionCtxGetDevResource", out_snapshot, [&](cudaDevResource& resource) {
        return cudaExecutionCtxGetDevResource(context->handle, &resource, static_cast<cudaDevResourceType>(resource_type));
    });
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)resource_type;
    if (out_snapshot != nullptr) *out_snapshot = JYPPX_CudaDevResourceSnapshot{};
    jyppx::cuda::set_cuda_error("cudaExecutionCtxGetDevResource", 0, "cuda-version-not-supported", "CUDA execution-context resource queries require CUDA Toolkit 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)resource_type;
    (void)out_snapshot;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context resource snapshot query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_dev_resource_snapshot_safe(
    JYPPX_CudaStream* stream,
    const int32_t resource_type,
    JYPPX_CudaDevResourceSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return query_dev_resource_snapshot("cudaStreamGetDevResource", out_snapshot, [&](cudaDevResource& resource) {
        return cudaStreamGetDevResource(stream_object->handle, &resource, static_cast<cudaDevResourceType>(resource_type));
    });
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)resource_type;
    if (out_snapshot != nullptr) *out_snapshot = JYPPX_CudaDevResourceSnapshot{};
    jyppx::cuda::set_cuda_error("cudaStreamGetDevResource", 0, "cuda-version-not-supported", "CUDA stream resource queries require CUDA Toolkit 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)resource_type;
    (void)out_snapshot;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream resource snapshot query");
#endif
}
