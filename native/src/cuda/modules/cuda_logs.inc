namespace
{
constexpr size_t kMaximumCudaLogBufferSize = 25600;

JYPPX_StatusCode cuda_logs_not_supported(const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "CUDA error log management requires CUDA runtime 13.2 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}
}

JYPPX_StatusCode jyppx_cuda_logs_get_current_cursor_safe(uint32_t* out_iterator)
{
    auto status = jyppx::cuda::validate_output_pointer(out_iterator, "out_iterator");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_iterator = 0;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13020
    cudaLogIterator iterator = 0;
    status = jyppx::cuda::map_cuda_status(cudaLogsCurrent(&iterator, 0U), "cudaLogsCurrent");
    if (status == JYPPX_STATUS_OK) { *out_iterator = static_cast<uint32_t>(iterator); }
    return status;
#elif JYPPX_HAS_CUDA_TOOLKIT
    return cuda_logs_not_supported("cudaLogsCurrent");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA log cursor query");
#endif
}

JYPPX_StatusCode jyppx_cuda_logs_dump_to_memory_safe(
    JYPPX_Boolean use_iterator,
    uint32_t iterator,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_written_size,
    uint32_t* out_next_iterator)
{
    auto status = jyppx::cuda::validate_output_pointer(out_written_size, "out_written_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_next_iterator, "out_next_iterator");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_written_size = 0;
    *out_next_iterator = 0;
    if (output_buffer == nullptr || output_buffer_size == 0 || output_buffer_size > kMaximumCudaLogBufferSize)
    {
        jyppx::cuda::set_cuda_error("cudaLogsDumpToMemory", 0, "invalid-log-buffer", "CUDA log output buffer size must be in the range [1, 25600].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    output_buffer[0] = '\0';
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13020
    cudaLogIterator next_iterator = static_cast<cudaLogIterator>(iterator);
    cudaLogIterator* iterator_pointer = use_iterator != JYPPX_FALSE ? &next_iterator : nullptr;
    size_t written_size = output_buffer_size;
    status = jyppx::cuda::map_cuda_status(
        cudaLogsDumpToMemory(iterator_pointer, output_buffer, &written_size, 0U),
        "cudaLogsDumpToMemory");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (written_size >= output_buffer_size)
    {
        output_buffer[0] = '\0';
        jyppx::cuda::set_cuda_error("cudaLogsDumpToMemory", 0, "invalid-log-size", "CUDA returned a log byte count that does not leave room for the required null terminator.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    output_buffer[written_size] = '\0';
    *out_written_size = written_size;
    *out_next_iterator = iterator_pointer != nullptr ? static_cast<uint32_t>(next_iterator) : 0U;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)use_iterator;
    (void)iterator;
    return cuda_logs_not_supported("cudaLogsDumpToMemory");
#else
    (void)use_iterator;
    (void)iterator;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA log memory dump");
#endif
}

JYPPX_StatusCode jyppx_cuda_logs_dump_to_file_safe(
    JYPPX_Boolean use_iterator,
    uint32_t iterator,
    const char* path,
    uint32_t* out_next_iterator)
{
    auto status = jyppx::cuda::validate_output_pointer(out_next_iterator, "out_next_iterator");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_next_iterator = 0;
    if (path == nullptr || path[0] == '\0')
    {
        jyppx::cuda::set_cuda_error("cudaLogsDumpToFile", 0, "invalid-log-path", "CUDA log output path must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13020
    cudaLogIterator next_iterator = static_cast<cudaLogIterator>(iterator);
    cudaLogIterator* iterator_pointer = use_iterator != JYPPX_FALSE ? &next_iterator : nullptr;
    status = jyppx::cuda::map_cuda_status(cudaLogsDumpToFile(iterator_pointer, path, 0U), "cudaLogsDumpToFile");
    if (status == JYPPX_STATUS_OK && iterator_pointer != nullptr)
    {
        *out_next_iterator = static_cast<uint32_t>(next_iterator);
    }
    return status;
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)use_iterator;
    (void)iterator;
    return cuda_logs_not_supported("cudaLogsDumpToFile");
#else
    (void)use_iterator;
    (void)iterator;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA log file dump");
#endif
}
