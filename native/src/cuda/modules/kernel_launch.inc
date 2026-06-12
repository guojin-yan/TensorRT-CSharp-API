namespace
{
JYPPX_StatusCode validate_cuda_function_pointer(const void* function, const char* operation)
{
    if (function == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-function", "CUDA function pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_cuda_dim3(const JYPPX_CudaDim3& value, const char* operation, const char* role)
{
    if (value.x == 0 || value.y == 0 || value.z == 0)
    {
        const std::string message = std::string(role) + " dimensions must all be greater than zero.";
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-dimensions", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_cuda_block_size(const int32_t block_size, const char* operation)
{
    if (block_size <= 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-block-size", "CUDA occupancy block size must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT
dim3 make_cuda_dim3(const JYPPX_CudaDim3& value)
{
    return dim3(value.x, value.y, value.z);
}

void copy_cuda_func_attributes(const cudaFuncAttributes& source, JYPPX_CudaFuncAttributes* destination)
{
    destination->shared_size_bytes = source.sharedSizeBytes;
    destination->const_size_bytes = source.constSizeBytes;
    destination->local_size_bytes = source.localSizeBytes;
    destination->max_threads_per_block = source.maxThreadsPerBlock;
    destination->num_registers = source.numRegs;
    destination->ptx_version = source.ptxVersion;
    destination->binary_version = source.binaryVersion;
    destination->cache_mode_ca = source.cacheModeCA;
    destination->max_dynamic_shared_size_bytes = source.maxDynamicSharedSizeBytes;
    destination->preferred_shared_memory_carveout = source.preferredShmemCarveout;
    destination->cluster_dim_must_be_set = 0;
    destination->required_cluster_width = 0;
    destination->required_cluster_height = 0;
    destination->required_cluster_depth = 0;
    destination->cluster_scheduling_policy_preference = 0;
    destination->non_portable_cluster_size_allowed = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    destination->cluster_dim_must_be_set = source.clusterDimMustBeSet;
    destination->required_cluster_width = source.requiredClusterWidth;
    destination->required_cluster_height = source.requiredClusterHeight;
    destination->required_cluster_depth = source.requiredClusterDepth;
    destination->cluster_scheduling_policy_preference = source.clusterSchedulingPolicyPreference;
    destination->non_portable_cluster_size_allowed = source.nonPortableClusterSizeAllowed;
#endif
}

JYPPX_StatusCode make_cuda_launch_config(const JYPPX_CudaLaunchConfig& source, cudaLaunchConfig_t* destination, const char* operation)
{
    auto status = validate_cuda_dim3(source.grid_dim, operation, "Grid");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_dim3(source.block_dim, operation, "Block");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    destination->gridDim = make_cuda_dim3(source.grid_dim);
    destination->blockDim = make_cuda_dim3(source.block_dim);
    destination->dynamicSmemBytes = source.dynamic_shared_memory_bytes;
    destination->stream = nullptr;
    destination->attrs = reinterpret_cast<cudaLaunchAttribute*>(source.attributes);
    destination->numAttrs = source.attribute_count;

    if (source.stream != nullptr)
    {
        status = jyppx::cuda::validate_stream(source.stream, "stream");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        destination->stream = reinterpret_cast<StreamObject*>(source.stream)->handle;
    }

    if (source.attribute_count > 0 && source.attributes == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-launch-attributes", "CUDA launch attributes must not be null when attribute count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_cuda_func_get_attributes(const void* function, JYPPX_CudaFuncAttributes* out_attributes)
{
    auto status = jyppx::cuda::validate_output_pointer(out_attributes, "out_attributes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_attributes, 0, sizeof(*out_attributes));
    status = validate_cuda_function_pointer(function, "cudaFuncGetAttributes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaFuncAttributes attributes{};
    status = jyppx::cuda::map_cuda_status(cudaFuncGetAttributes(&attributes, function), "cudaFuncGetAttributes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_cuda_func_attributes(attributes, out_attributes);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_get_name(const void* function, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = validate_cuda_function_pointer(function, "cudaFuncGetName");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    const char* name = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaFuncGetName(&name, function), "cudaFuncGetName");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)output_buffer;
    (void)output_buffer_size;
    (void)out_required_size;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function name query");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_get_param_count(const void* function, size_t* out_param_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_param_count, "out_param_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_param_count = 0;
    status = validate_cuda_function_pointer(function, "cudaFuncGetParamCount");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaFuncGetParamCount(function, out_param_count), "cudaFuncGetParamCount");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function parameter count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_get_param_info(const void* function, size_t parameter_index, size_t* out_parameter_offset, size_t* out_parameter_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_parameter_offset, "out_parameter_offset");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(out_parameter_size, "out_parameter_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parameter_offset = 0;
    *out_parameter_size = 0;
    status = validate_cuda_function_pointer(function, "cudaFuncGetParamInfo");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    return jyppx::cuda::map_cuda_status(cudaFuncGetParamInfo(function, parameter_index, out_parameter_offset, out_parameter_size), "cudaFuncGetParamInfo");
#else
    (void)parameter_index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function parameter info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_set_attribute(const void* function, int32_t attribute, int32_t value)
{
    auto status = validate_cuda_function_pointer(function, "cudaFuncSetAttribute");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaFuncSetAttribute(function, static_cast<cudaFuncAttribute>(attribute), value), "cudaFuncSetAttribute");
#else
    (void)attribute;
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function attribute set");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_set_cache_config(const void* function, int32_t cache_config)
{
    auto status = validate_cuda_function_pointer(function, "cudaFuncSetCacheConfig");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaFuncSetCacheConfig(function, static_cast<cudaFuncCache>(cache_config)), "cudaFuncSetCacheConfig");
#else
    (void)cache_config;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function cache config set");
#endif
}

JYPPX_StatusCode jyppx_cuda_func_set_shared_mem_config(const void* function, int32_t config)
{
    auto status = validate_cuda_function_pointer(function, "cudaFuncSetSharedMemConfig");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaFuncSetSharedMemConfig(function, static_cast<cudaSharedMemConfig>(config)), "cudaFuncSetSharedMemConfig");
#else
    (void)config;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA function shared memory config set");
#endif
}

JYPPX_StatusCode jyppx_cuda_launch_kernel(
    const void* function,
    JYPPX_CudaDim3 grid_dim,
    JYPPX_CudaDim3 block_dim,
    void** arguments,
    size_t shared_memory_bytes,
    JYPPX_CudaStream* stream)
{
    auto status = validate_cuda_function_pointer(function, "cudaLaunchKernel");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_dim3(grid_dim, "cudaLaunchKernel", "Grid");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_dim3(block_dim, "cudaLaunchKernel", "Block");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaLaunchKernel(function, make_cuda_dim3(grid_dim), make_cuda_dim3(block_dim), arguments, shared_memory_bytes, stream_object->handle),
        "cudaLaunchKernel");
#else
    (void)arguments;
    (void)shared_memory_bytes;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel launch");
#endif
}

JYPPX_StatusCode jyppx_cuda_launch_kernel_ex(const JYPPX_CudaLaunchConfig* config, const void* function, void** arguments)
{
    if (config == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaLaunchKernelExC", 0, "invalid-config", "CUDA launch config must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_cuda_function_pointer(function, "cudaLaunchKernelExC");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaLaunchConfig_t launch_config{};
    status = make_cuda_launch_config(*config, &launch_config, "cudaLaunchKernelExC");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx::cuda::map_cuda_status(cudaLaunchKernelExC(&launch_config, function, arguments), "cudaLaunchKernelExC");
#else
    (void)arguments;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA extended kernel launch");
#endif
}

JYPPX_StatusCode jyppx_cuda_launch_cooperative_kernel(
    const void* function,
    JYPPX_CudaDim3 grid_dim,
    JYPPX_CudaDim3 block_dim,
    void** arguments,
    size_t shared_memory_bytes,
    JYPPX_CudaStream* stream)
{
    auto status = validate_cuda_function_pointer(function, "cudaLaunchCooperativeKernel");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_dim3(grid_dim, "cudaLaunchCooperativeKernel", "Grid");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_dim3(block_dim, "cudaLaunchCooperativeKernel", "Block");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaLaunchCooperativeKernel(function, make_cuda_dim3(grid_dim), make_cuda_dim3(block_dim), arguments, shared_memory_bytes, stream_object->handle),
        "cudaLaunchCooperativeKernel");
#else
    (void)arguments;
    (void)shared_memory_bytes;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA cooperative kernel launch");
#endif
}

JYPPX_StatusCode jyppx_cuda_occupancy_available_dynamic_smem_per_block(size_t* out_dynamic_shared_memory_size, const void* function, int32_t num_blocks, int32_t block_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_dynamic_shared_memory_size, "out_dynamic_shared_memory_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_dynamic_shared_memory_size = 0;
    status = validate_cuda_function_pointer(function, "cudaOccupancyAvailableDynamicSMemPerBlock");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (num_blocks <= 0)
    {
        jyppx::cuda::set_cuda_error("cudaOccupancyAvailableDynamicSMemPerBlock", 0, "invalid-block-count", "CUDA occupancy block count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_cuda_block_size(block_size, "cudaOccupancyAvailableDynamicSMemPerBlock");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaOccupancyAvailableDynamicSMemPerBlock(out_dynamic_shared_memory_size, function, num_blocks, block_size), "cudaOccupancyAvailableDynamicSMemPerBlock");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA occupancy dynamic shared memory query");
#endif
}

JYPPX_StatusCode jyppx_cuda_occupancy_max_active_blocks_per_multiprocessor(int32_t* out_num_blocks, const void* function, int32_t block_size, size_t dynamic_shared_memory_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_num_blocks, "out_num_blocks");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_num_blocks = 0;
    status = validate_cuda_function_pointer(function, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_block_size(block_size, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaOccupancyMaxActiveBlocksPerMultiprocessor(out_num_blocks, function, block_size, dynamic_shared_memory_size), "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA occupancy active block query");
#endif
}

JYPPX_StatusCode jyppx_cuda_occupancy_max_active_blocks_per_multiprocessor_with_flags(int32_t* out_num_blocks, const void* function, int32_t block_size, size_t dynamic_shared_memory_size, uint32_t flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_num_blocks, "out_num_blocks");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_num_blocks = 0;
    status = validate_cuda_function_pointer(function, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_block_size(block_size, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out_num_blocks, function, block_size, dynamic_shared_memory_size, flags), "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA occupancy active block query with flags");
#endif
}

JYPPX_StatusCode jyppx_cuda_occupancy_max_potential_cluster_size(int32_t* out_cluster_size, const void* function, const JYPPX_CudaLaunchConfig* config)
{
    auto status = jyppx::cuda::validate_output_pointer(out_cluster_size, "out_cluster_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cluster_size = 0;
    if (config == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaOccupancyMaxPotentialClusterSize", 0, "invalid-config", "CUDA launch config must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_cuda_function_pointer(function, "cudaOccupancyMaxPotentialClusterSize");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaLaunchConfig_t launch_config{};
    status = make_cuda_launch_config(*config, &launch_config, "cudaOccupancyMaxPotentialClusterSize");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx::cuda::map_cuda_status(cudaOccupancyMaxPotentialClusterSize(out_cluster_size, function, &launch_config), "cudaOccupancyMaxPotentialClusterSize");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA occupancy cluster size query");
#endif
}

JYPPX_StatusCode jyppx_cuda_occupancy_max_active_clusters(int32_t* out_num_clusters, const void* function, const JYPPX_CudaLaunchConfig* config)
{
    auto status = jyppx::cuda::validate_output_pointer(out_num_clusters, "out_num_clusters");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_num_clusters = 0;
    if (config == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaOccupancyMaxActiveClusters", 0, "invalid-config", "CUDA launch config must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_cuda_function_pointer(function, "cudaOccupancyMaxActiveClusters");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaLaunchConfig_t launch_config{};
    status = make_cuda_launch_config(*config, &launch_config, "cudaOccupancyMaxActiveClusters");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx::cuda::map_cuda_status(cudaOccupancyMaxActiveClusters(out_num_clusters, function, &launch_config), "cudaOccupancyMaxActiveClusters");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA occupancy active cluster query");
#endif
}
