JYPPX_StatusCode jyppx_cuda_runtime_get_version(int32_t* out_version)
{
    return jyppx_cuda_get_runtime_version(out_version);
}

JYPPX_StatusCode jyppx_cuda_driver_get_version(int32_t* out_version)
{
    return jyppx_cuda_get_driver_version(out_version);
}

JYPPX_StatusCode jyppx_cuda_device_get_count(int32_t* out_device_count)
{
    return jyppx_cuda_get_device_count(out_device_count);
}

JYPPX_StatusCode jyppx_cuda_device_get_properties(int32_t ordinal, JYPPX_CudaDeviceInfo* out_info)
{
    return jyppx_cuda_get_device_info(ordinal, out_info);
}

JYPPX_StatusCode jyppx_cuda_device_get_current(int32_t* out_device)
{
    return jyppx_cuda_get_current_device(out_device);
}

JYPPX_StatusCode jyppx_cuda_device_set_current(int32_t device)
{
    return jyppx_cuda_set_device(device);
}

JYPPX_StatusCode jyppx_cuda_device_synchronize_v2(void)
{
    return jyppx_cuda_device_synchronize();
}

JYPPX_StatusCode jyppx_cuda_device_reset_v2(void)
{
    return jyppx_cuda_device_reset();
}

JYPPX_StatusCode jyppx_cuda_device_get_attribute_v2(int32_t device, int32_t attribute, int32_t* out_value)
{
    return jyppx_cuda_device_get_attribute(device, attribute, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_limit_v2(int32_t limit, size_t* out_value)
{
    return jyppx_cuda_device_get_limit(limit, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_set_limit_v2(int32_t limit, size_t value)
{
    return jyppx_cuda_device_set_limit(limit, value);
}

JYPPX_StatusCode jyppx_cuda_device_get_cache_config_v2(int32_t* out_config)
{
    return jyppx_cuda_device_get_cache_config(out_config);
}

JYPPX_StatusCode jyppx_cuda_device_set_cache_config_v2(int32_t config)
{
    return jyppx_cuda_device_set_cache_config(config);
}

JYPPX_StatusCode jyppx_cuda_device_get_pci_bus_id_v2(int32_t device, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_cuda_device_get_pci_bus_id(device, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_cuda_device_get_by_pci_bus_id_v2(const char* pci_bus_id, int32_t* out_device)
{
    return jyppx_cuda_device_get_by_pci_bus_id(pci_bus_id, out_device);
}

JYPPX_StatusCode jyppx_cuda_device_can_access_peer_v2(int32_t device, int32_t peer_device, JYPPX_Boolean* out_can_access)
{
    return jyppx_cuda_device_can_access_peer(device, peer_device, out_can_access);
}

JYPPX_StatusCode jyppx_cuda_device_enable_peer_access_v2(int32_t peer_device, uint32_t flags)
{
    return jyppx_cuda_device_enable_peer_access(peer_device, flags);
}

JYPPX_StatusCode jyppx_cuda_device_disable_peer_access_v2(int32_t peer_device)
{
    return jyppx_cuda_device_disable_peer_access(peer_device);
}

JYPPX_StatusCode jyppx_cuda_device_get_p2p_attribute_v2(int32_t attribute, int32_t source_device, int32_t destination_device, int32_t* out_value)
{
    return jyppx_cuda_device_get_p2p_attribute(attribute, source_device, destination_device, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_flags_v2(uint32_t* out_flags)
{
    return jyppx_cuda_get_device_flags(out_flags);
}

JYPPX_StatusCode jyppx_cuda_device_set_flags_v2(uint32_t flags)
{
    return jyppx_cuda_set_device_flags(flags);
}

JYPPX_StatusCode jyppx_cuda_mem_get_info(JYPPX_CudaMemoryInfo* out_info)
{
    return jyppx_cuda_get_memory_info(out_info);
}

JYPPX_StatusCode jyppx_cuda_malloc(size_t size, JYPPX_CudaMemory** out_memory)
{
    return jyppx_cuda_memory_alloc(size, out_memory);
}

JYPPX_StatusCode jyppx_cuda_malloc_managed(size_t size, uint32_t flags, JYPPX_CudaMemory** out_memory)
{
    return jyppx_cuda_memory_alloc_managed(size, flags, out_memory);
}

JYPPX_StatusCode jyppx_cuda_malloc_async(size_t size, JYPPX_CudaStream* stream, JYPPX_CudaMemory** out_memory)
{
    return jyppx_cuda_memory_alloc_async(size, stream, out_memory);
}

JYPPX_StatusCode jyppx_cuda_malloc_from_pool_async(size_t size, uint64_t pool_handle, JYPPX_CudaStream* stream, JYPPX_CudaMemory** out_memory)
{
    return jyppx_cuda_memory_alloc_from_pool_async(size, pool_handle, stream, out_memory);
}

JYPPX_StatusCode jyppx_cuda_mem_get_size(JYPPX_CudaMemory* memory, size_t* out_size)
{
    return jyppx_cuda_memory_get_size(memory, out_size);
}

JYPPX_StatusCode jyppx_cuda_mem_get_device_pointer(JYPPX_CudaMemory* memory, void** out_pointer)
{
    return jyppx_cuda_memory_get_device_pointer(memory, out_pointer);
}

JYPPX_StatusCode jyppx_cuda_pointer_get_attributes(JYPPX_CudaMemory* memory, JYPPX_CudaPointerAttributes* out_attributes)
{
    return jyppx_cuda_memory_get_pointer_attributes(memory, out_attributes);
}

JYPPX_StatusCode jyppx_cuda_memcpy_host_to_device(JYPPX_CudaMemory* memory, const void* source, size_t size)
{
    return jyppx_cuda_memory_copy_from_host(memory, source, size);
}

JYPPX_StatusCode jyppx_cuda_memcpy_device_to_host(JYPPX_CudaMemory* memory, void* destination, size_t size)
{
    return jyppx_cuda_memory_copy_to_host(memory, destination, size);
}

JYPPX_StatusCode jyppx_cuda_memcpy_host_to_device_async(JYPPX_CudaMemory* memory, const void* source, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_copy_from_host_async(memory, source, size, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_device_to_host_async(JYPPX_CudaMemory* memory, void* destination, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_copy_to_host_async(memory, destination, size, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_device_to_device(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size)
{
    return jyppx_cuda_memory_copy_device_to_device(destination, source, size);
}

JYPPX_StatusCode jyppx_cuda_memcpy_device_to_device_async(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_copy_device_to_device_async(destination, source, size, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_default(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size)
{
    return jyppx_cuda_memory_copy_default(destination, source, size);
}

JYPPX_StatusCode jyppx_cuda_memcpy_default_async(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_copy_default_async(destination, source, size, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_peer(JYPPX_CudaMemory* destination, int32_t destination_device, JYPPX_CudaMemory* source, int32_t source_device, size_t size)
{
    return jyppx_cuda_memory_copy_peer(destination, destination_device, source, source_device, size);
}

JYPPX_StatusCode jyppx_cuda_memcpy_peer_async(JYPPX_CudaMemory* destination, int32_t destination_device, JYPPX_CudaMemory* source, int32_t source_device, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_copy_peer_async(destination, destination_device, source, source_device, size, stream);
}

JYPPX_StatusCode jyppx_cuda_memset(JYPPX_CudaMemory* memory, int32_t value, size_t size)
{
    return jyppx_cuda_memory_memset(memory, value, size);
}

JYPPX_StatusCode jyppx_cuda_memset_async(JYPPX_CudaMemory* memory, int32_t value, size_t size, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_memset_async(memory, value, size, stream);
}

JYPPX_StatusCode jyppx_cuda_mem_prefetch_async(JYPPX_CudaMemory* memory, size_t size, int32_t destination_device, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_prefetch_async(memory, size, destination_device, stream);
}

JYPPX_StatusCode jyppx_cuda_mem_advise(JYPPX_CudaMemory* memory, size_t size, int32_t advice, int32_t device)
{
    return jyppx_cuda_memory_advise(memory, size, advice, device);
}

JYPPX_StatusCode jyppx_cuda_free(JYPPX_CudaMemory* memory)
{
    return jyppx_cuda_memory_free(memory);
}

JYPPX_StatusCode jyppx_cuda_free_async(JYPPX_CudaMemory* memory, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_free_async(memory, stream);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_create(int32_t device, uint64_t* out_pool_handle)
{
    return jyppx_cuda_memory_pool_create(device, out_pool_handle);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_destroy(uint64_t pool_handle)
{
    return jyppx_cuda_memory_pool_destroy(pool_handle);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_get_attribute(uint64_t pool_handle, int32_t attribute, int64_t* out_value)
{
    return jyppx_cuda_memory_pool_get_attribute(pool_handle, attribute, out_value);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_set_attribute(uint64_t pool_handle, int32_t attribute, int64_t value)
{
    return jyppx_cuda_memory_pool_set_attribute(pool_handle, attribute, value);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_trim_to(uint64_t pool_handle, size_t min_bytes_to_keep)
{
    return jyppx_cuda_memory_pool_trim_to(pool_handle, min_bytes_to_keep);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_set_access(uint64_t pool_handle, int32_t device, uint32_t flags)
{
    return jyppx_cuda_memory_pool_set_access(pool_handle, device, flags);
}

JYPPX_StatusCode jyppx_cuda_mem_pool_get_access(uint64_t pool_handle, int32_t device, uint32_t* out_flags)
{
    return jyppx_cuda_memory_pool_get_access(pool_handle, device, out_flags);
}

JYPPX_StatusCode jyppx_cuda_malloc_pitch(size_t width_bytes, size_t height, JYPPX_CudaPitchedMemory** out_memory)
{
    return jyppx_cuda_pitched_memory_alloc(width_bytes, height, out_memory);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_host_to_device(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height)
{
    return jyppx_cuda_pitched_memory_copy_from_host_2d(memory, source, source_pitch, width_bytes, height);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_device_to_host(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height)
{
    return jyppx_cuda_pitched_memory_copy_to_host_2d(memory, destination, destination_pitch, width_bytes, height);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_device_to_device(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height)
{
    return jyppx_cuda_pitched_memory_copy_device_to_device_2d(destination, source, width_bytes, height);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_host_to_device_async(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_from_host_2d_async(memory, source, source_pitch, width_bytes, height, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_device_to_host_async(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_to_host_2d_async(memory, destination, destination_pitch, width_bytes, height, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_device_to_device_async(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_device_to_device_2d_async(destination, source, width_bytes, height, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_host_to_device(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, size_t depth)
{
    return jyppx_cuda_pitched_memory_copy_from_host_3d(memory, source, source_pitch, width_bytes, height, depth);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_device_to_host(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, size_t depth)
{
    return jyppx_cuda_pitched_memory_copy_to_host_3d(memory, destination, destination_pitch, width_bytes, height, depth);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_device_to_device(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, size_t depth)
{
    return jyppx_cuda_pitched_memory_copy_device_to_device_3d(destination, source, width_bytes, height, depth);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_host_to_device_async(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_from_host_3d_async(memory, source, source_pitch, width_bytes, height, depth, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_device_to_host_async(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_to_host_3d_async(memory, destination, destination_pitch, width_bytes, height, depth, stream);
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_device_to_device_async(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_copy_device_to_device_3d_async(destination, source, width_bytes, height, depth, stream);
}

JYPPX_StatusCode jyppx_cuda_memset_2d(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height)
{
    return jyppx_cuda_pitched_memory_memset_2d(memory, value, width_bytes, height);
}

JYPPX_StatusCode jyppx_cuda_memset_2d_async(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_memset_2d_async(memory, value, width_bytes, height, stream);
}

JYPPX_StatusCode jyppx_cuda_memset_3d(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, size_t depth)
{
    return jyppx_cuda_pitched_memory_memset_3d(memory, value, width_bytes, height, depth);
}

JYPPX_StatusCode jyppx_cuda_memset_3d_async(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_pitched_memory_memset_3d_async(memory, value, width_bytes, height, depth, stream);
}

JYPPX_StatusCode jyppx_cuda_free_pitch(JYPPX_CudaPitchedMemory* memory)
{
    return jyppx_cuda_pitched_memory_free(memory);
}

JYPPX_StatusCode jyppx_cuda_host_alloc(size_t size, JYPPX_CudaPinnedMemory** out_memory)
{
    return jyppx_cuda_pinned_memory_alloc(size, out_memory);
}

JYPPX_StatusCode jyppx_cuda_host_alloc_with_flags(size_t size, uint32_t flags, JYPPX_CudaPinnedMemory** out_memory)
{
    return jyppx_cuda_pinned_memory_alloc_with_flags(size, flags, out_memory);
}

JYPPX_StatusCode jyppx_cuda_host_register(void* pointer, size_t size, uint32_t flags, JYPPX_CudaPinnedMemory** out_memory)
{
    return jyppx_cuda_pinned_memory_register(pointer, size, flags, out_memory);
}

JYPPX_StatusCode jyppx_cuda_host_get_size(JYPPX_CudaPinnedMemory* memory, size_t* out_size)
{
    return jyppx_cuda_pinned_memory_get_size(memory, out_size);
}

JYPPX_StatusCode jyppx_cuda_host_get_flags(JYPPX_CudaPinnedMemory* memory, uint32_t* out_flags)
{
    return jyppx_cuda_pinned_memory_get_flags(memory, out_flags);
}

JYPPX_StatusCode jyppx_cuda_host_get_pointer(JYPPX_CudaPinnedMemory* memory, void** out_pointer)
{
    return jyppx_cuda_pinned_memory_get_host_pointer(memory, out_pointer);
}

JYPPX_StatusCode jyppx_cuda_host_get_device_pointer(JYPPX_CudaPinnedMemory* memory, void** out_pointer)
{
    return jyppx_cuda_pinned_memory_get_mapped_device_pointer(memory, out_pointer);
}

JYPPX_StatusCode jyppx_cuda_free_host(JYPPX_CudaPinnedMemory* memory)
{
    return jyppx_cuda_pinned_memory_free(memory);
}

JYPPX_StatusCode jyppx_cuda_stream_create_v2(uint32_t flags, JYPPX_CudaStream** out_stream)
{
    return jyppx_cuda_stream_create(flags, out_stream);
}

JYPPX_StatusCode jyppx_cuda_stream_create_with_priority_v2(uint32_t flags, int32_t priority, JYPPX_CudaStream** out_stream)
{
    return jyppx_cuda_stream_create_with_priority(flags, priority, out_stream);
}

JYPPX_StatusCode jyppx_cuda_stream_get_priority_range_v2(int32_t* out_least_priority, int32_t* out_greatest_priority)
{
    return jyppx_cuda_stream_get_priority_range(out_least_priority, out_greatest_priority);
}

JYPPX_StatusCode jyppx_cuda_stream_get_flags_v2(JYPPX_CudaStream* stream, uint32_t* out_flags)
{
    return jyppx_cuda_stream_get_flags(stream, out_flags);
}

JYPPX_StatusCode jyppx_cuda_stream_get_priority_v2(JYPPX_CudaStream* stream, int32_t* out_priority)
{
    return jyppx_cuda_stream_get_priority(stream, out_priority);
}

JYPPX_StatusCode jyppx_cuda_stream_get_device_v2(JYPPX_CudaStream* stream, int32_t* out_device)
{
    return jyppx_cuda_stream_get_device(stream, out_device);
}

JYPPX_StatusCode jyppx_cuda_stream_get_id_v2(JYPPX_CudaStream* stream, uint64_t* out_stream_id)
{
    return jyppx_cuda_stream_get_id(stream, out_stream_id);
}

JYPPX_StatusCode jyppx_cuda_stream_query_v2(JYPPX_CudaStream* stream)
{
    return jyppx_cuda_stream_query(stream);
}

JYPPX_StatusCode jyppx_cuda_stream_synchronize_v2(JYPPX_CudaStream* stream)
{
    return jyppx_cuda_stream_synchronize(stream);
}

JYPPX_StatusCode jyppx_cuda_stream_wait_event_v2(JYPPX_CudaStream* stream, JYPPX_CudaEvent* event_handle, uint32_t flags)
{
    return jyppx_cuda_stream_wait_event(stream, event_handle, flags);
}

JYPPX_StatusCode jyppx_cuda_stream_begin_capture_v2(JYPPX_CudaStream* stream, int32_t mode)
{
    return jyppx_cuda_stream_begin_capture(stream, mode);
}

JYPPX_StatusCode jyppx_cuda_stream_end_capture_v2(JYPPX_CudaStream* stream, JYPPX_CudaGraph** out_graph)
{
    return jyppx_cuda_stream_end_capture(stream, out_graph);
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_info_v2(JYPPX_CudaStream* stream, int32_t* out_capture_status, uint64_t* out_capture_id)
{
    return jyppx_cuda_stream_get_capture_info(stream, out_capture_status, out_capture_id);
}

JYPPX_StatusCode jyppx_cuda_stream_is_capturing_v2(JYPPX_CudaStream* stream, int32_t* out_capture_status)
{
    return jyppx_cuda_stream_is_capturing(stream, out_capture_status);
}

JYPPX_StatusCode jyppx_cuda_stream_copy_attributes_v2(JYPPX_CudaStream* destination, JYPPX_CudaStream* source)
{
    return jyppx_cuda_stream_copy_attributes(destination, source);
}

JYPPX_StatusCode jyppx_cuda_stream_destroy_v2(JYPPX_CudaStream* stream)
{
    return jyppx_cuda_stream_destroy(stream);
}

JYPPX_StatusCode jyppx_cuda_event_create_v2(uint32_t flags, JYPPX_CudaEvent** out_event)
{
    return jyppx_cuda_event_create(flags, out_event);
}

JYPPX_StatusCode jyppx_cuda_event_create_with_flags_v2(uint32_t flags, JYPPX_CudaEvent** out_event)
{
    return jyppx_cuda_event_create(flags, out_event);
}

JYPPX_StatusCode jyppx_cuda_event_record_v2(JYPPX_CudaEvent* event_handle, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_event_record(event_handle, stream);
}

JYPPX_StatusCode jyppx_cuda_event_record_with_flags_v2(JYPPX_CudaEvent* event_handle, JYPPX_CudaStream* stream, uint32_t flags)
{
    return jyppx_cuda_event_record_with_flags(event_handle, stream, flags);
}

JYPPX_StatusCode jyppx_cuda_event_synchronize_v2(JYPPX_CudaEvent* event_handle)
{
    return jyppx_cuda_event_synchronize(event_handle);
}

JYPPX_StatusCode jyppx_cuda_event_query_v2(JYPPX_CudaEvent* event_handle)
{
    return jyppx_cuda_event_query(event_handle);
}

JYPPX_StatusCode jyppx_cuda_event_get_flags_v2(JYPPX_CudaEvent* event_handle, uint32_t* out_flags)
{
    return jyppx_cuda_event_get_flags(event_handle, out_flags);
}

JYPPX_StatusCode jyppx_cuda_event_elapsed_time_official(JYPPX_CudaEvent* start_event, JYPPX_CudaEvent* end_event, float* out_milliseconds)
{
    return jyppx_cuda_event_elapsed_time(start_event, end_event, out_milliseconds);
}

JYPPX_StatusCode jyppx_cuda_event_destroy_v2(JYPPX_CudaEvent* event_handle)
{
    return jyppx_cuda_event_destroy(event_handle);
}

JYPPX_StatusCode jyppx_cuda_graph_create_v2(uint32_t flags, JYPPX_CudaGraph** out_graph)
{
    return jyppx_cuda_graph_create(flags, out_graph);
}

JYPPX_StatusCode jyppx_cuda_graph_clone_v2(JYPPX_CudaGraph* graph, JYPPX_CudaGraph** out_clone)
{
    return jyppx_cuda_graph_clone(graph, out_clone);
}

JYPPX_StatusCode jyppx_cuda_graph_get_node_count_v2(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_node_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_get_root_node_count_v2(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_root_node_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_get_edge_count_v2(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_edge_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_instantiate_v2(JYPPX_CudaGraph* graph, uint64_t flags, JYPPX_CudaGraphExec** out_graph_exec)
{
    return jyppx_cuda_graph_instantiate(graph, flags, out_graph_exec);
}

JYPPX_StatusCode jyppx_cuda_graph_exec_launch_v2(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_graph_exec_launch(graph_exec, stream);
}

JYPPX_StatusCode jyppx_cuda_graph_exec_upload_v2(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_graph_exec_upload(graph_exec, stream);
}

JYPPX_StatusCode jyppx_cuda_graph_exec_get_flags_v2(JYPPX_CudaGraphExec* graph_exec, uint64_t* out_flags)
{
    return jyppx_cuda_graph_exec_get_flags(graph_exec, out_flags);
}

JYPPX_StatusCode jyppx_cuda_graph_exec_destroy_v2(JYPPX_CudaGraphExec* graph_exec)
{
    return jyppx_cuda_graph_exec_destroy(graph_exec);
}

JYPPX_StatusCode jyppx_cuda_graph_destroy_v2(JYPPX_CudaGraph* graph)
{
    return jyppx_cuda_graph_destroy(graph);
}
