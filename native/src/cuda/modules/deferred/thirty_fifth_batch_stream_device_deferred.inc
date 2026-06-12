JYPPX_StatusCode jyppx_cuda_choose_device_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaChooseDevice", 0, "deferred-interface-boundary", "cudaChooseDevice deferred: cudaDeviceProp parity is not modeled by the current bridge ABI");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_device_get_dev_resource_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaDeviceGetDevResource", 0, "deferred-interface-boundary", "cudaDeviceGetDevResource deferred: CUDA 13 device resource union ownership is not modeled by the current bridge ABI");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_device_get_execution_ctx_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaDeviceGetExecutionCtx", 0, "deferred-interface-boundary", "cudaDeviceGetExecutionCtx deferred: CUDA execution context lifecycle requires a dedicated bridge object model");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_device_get_nv_sci_sync_attributes_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaDeviceGetNvSciSyncAttributes", 0, "deferred-interface-boundary", "cudaDeviceGetNvSciSyncAttributes deferred: NvSci external-resource ABI is not modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_device_register_async_notification_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaDeviceRegisterAsyncNotification", 0, "deferred-interface-boundary", "cudaDeviceRegisterAsyncNotification deferred: native callback lifetime is not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_device_unregister_async_notification_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaDeviceUnregisterAsyncNotification", 0, "deferred-interface-boundary", "cudaDeviceUnregisterAsyncNotification deferred: native callback handle lifetime is not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_add_callback_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamAddCallback", 0, "deferred-interface-boundary", "cudaStreamAddCallback deferred: native callback lifetime is not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_begin_capture_to_graph_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamBeginCaptureToGraph", 0, "deferred-interface-boundary", "cudaStreamBeginCaptureToGraph deferred: graph node and edge-data arrays are not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_info_ptsz_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamGetCaptureInfo_ptsz", 0, "deferred-interface-boundary", "cudaStreamGetCaptureInfo_ptsz deferred: per-thread default stream variant is not separately modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_info_v3_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamGetCaptureInfo_v3", 0, "deferred-interface-boundary", "cudaStreamGetCaptureInfo_v3 deferred: graph node and edge-data output arrays are not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_get_dev_resource_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamGetDevResource", 0, "deferred-interface-boundary", "cudaStreamGetDevResource deferred: CUDA 13 device resource union ownership is not modeled by the current bridge ABI");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies", 0, "deferred-interface-boundary", "cudaStreamUpdateCaptureDependencies deferred: graph node dependency arrays are not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_ptsz_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_ptsz", 0, "deferred-interface-boundary", "cudaStreamUpdateCaptureDependencies_ptsz deferred: per-thread default stream variant is not separately modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_v2_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_v2", 0, "deferred-interface-boundary", "cudaStreamUpdateCaptureDependencies_v2 deferred: graph node edge-data arrays are not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_launch_host_func_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaLaunchHostFunc", 0, "deferred-interface-boundary", "cudaLaunchHostFunc deferred: native callback lifetime is not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}

JYPPX_StatusCode jyppx_cuda_launch_host_func_v2_deferred(void)
{
    jyppx::cuda::set_cuda_error("cudaLaunchHostFunc_v2", 0, "deferred-interface-boundary", "cudaLaunchHostFunc_v2 deferred: native callback lifetime is not safely modeled by the current bridge");
    return JYPPX_STATUS_NOT_IMPLEMENTED;
}
