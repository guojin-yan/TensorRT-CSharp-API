namespace
{
void clear_resource_descriptor_snapshot(JYPPX_CudaResourceDescriptorSnapshot* snapshot)
{
    std::memset(snapshot, 0, sizeof(*snapshot));
}

void clear_texture_descriptor(JYPPX_CudaTextureDescriptor* descriptor)
{
    std::memset(descriptor, 0, sizeof(*descriptor));
}

void clear_texture_resource_view_snapshot(JYPPX_CudaTextureResourceViewSnapshot* snapshot)
{
    std::memset(snapshot, 0, sizeof(*snapshot));
}

JYPPX_StatusCode report_texture_surface_exception(const char* operation, const char* error_name, const char* message) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(operation, 0, error_name, message);
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_RUNTIME_ERROR;
}

template <typename TCallable>
JYPPX_StatusCode run_texture_surface_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return report_texture_surface_exception(operation, "native-allocation-failed", "The CUDA texture/surface bridge could not allocate native state.");
    }
    catch (...)
    {
        return report_texture_surface_exception(operation, "native-exception-caught", "A native exception was contained by the CUDA texture/surface bridge.");
    }
}

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode validate_texture_descriptor(
    const JYPPX_CudaTextureDescriptor* descriptor,
    const bool supports_seamless_cubemap,
    const char* operation)
{
    if (descriptor == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "missing-texture-descriptor", "The CUDA texture descriptor must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const auto valid_address_mode = [](const int32_t value) { return value >= 0 && value <= 3; };
    if (!valid_address_mode(descriptor->address_mode_x) ||
        !valid_address_mode(descriptor->address_mode_y) ||
        !valid_address_mode(descriptor->address_mode_z))
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-address-mode", "CUDA texture address modes must be Wrap, Clamp, Mirror, or Border.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->filter_mode < 0 || descriptor->filter_mode > 1 ||
        descriptor->mipmap_filter_mode < 0 || descriptor->mipmap_filter_mode > 1)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-filter-mode", "CUDA texture filter modes must be Point or Linear.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->read_mode < 0 || descriptor->read_mode > 1)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-read-mode", "CUDA texture read mode must be ElementType or NormalizedFloat.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->max_anisotropy > 16U)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-max-anisotropy", "CUDA texture max anisotropy must be between zero and sixteen.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->min_mipmap_level_clamp > descriptor->max_mipmap_level_clamp)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-mipmap-clamp", "CUDA texture minimum mipmap clamp must not exceed the maximum clamp.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->seamless_cubemap != JYPPX_FALSE && !supports_seamless_cubemap)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "seamless-cubemap-not-supported", "This CUDA texture descriptor ABI does not expose seamless cubemap filtering.");
        return JYPPX_STATUS_NOT_SUPPORTED;
    }

    return JYPPX_STATUS_OK;
}

void fill_texture_descriptor_common(const JYPPX_CudaTextureDescriptor& source, cudaTextureDesc* destination)
{
    destination->addressMode[0] = static_cast<cudaTextureAddressMode>(source.address_mode_x);
    destination->addressMode[1] = static_cast<cudaTextureAddressMode>(source.address_mode_y);
    destination->addressMode[2] = static_cast<cudaTextureAddressMode>(source.address_mode_z);
    destination->filterMode = static_cast<cudaTextureFilterMode>(source.filter_mode);
    destination->readMode = static_cast<cudaTextureReadMode>(source.read_mode);
    destination->sRGB = source.srgb != JYPPX_FALSE ? 1 : 0;
    destination->borderColor[0] = source.border_color_r;
    destination->borderColor[1] = source.border_color_g;
    destination->borderColor[2] = source.border_color_b;
    destination->borderColor[3] = source.border_color_a;
    destination->normalizedCoords = source.normalized_coordinates != JYPPX_FALSE ? 1 : 0;
    destination->maxAnisotropy = source.max_anisotropy;
    destination->mipmapFilterMode = static_cast<cudaTextureFilterMode>(source.mipmap_filter_mode);
    destination->mipmapLevelBias = source.mipmap_level_bias;
    destination->minMipmapLevelClamp = source.min_mipmap_level_clamp;
    destination->maxMipmapLevelClamp = source.max_mipmap_level_clamp;
    destination->disableTrilinearOptimization = source.disable_trilinear_optimization != JYPPX_FALSE ? 1 : 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    destination->seamlessCubemap = source.seamless_cubemap != JYPPX_FALSE ? 1 : 0;
#endif
}

void copy_texture_descriptor_common(const cudaTextureDesc& source, JYPPX_CudaTextureDescriptor* destination)
{
    destination->address_mode_x = static_cast<int32_t>(source.addressMode[0]);
    destination->address_mode_y = static_cast<int32_t>(source.addressMode[1]);
    destination->address_mode_z = static_cast<int32_t>(source.addressMode[2]);
    destination->filter_mode = static_cast<int32_t>(source.filterMode);
    destination->read_mode = static_cast<int32_t>(source.readMode);
    destination->srgb = to_jyppx_bool(source.sRGB != 0);
    destination->border_color_r = source.borderColor[0];
    destination->border_color_g = source.borderColor[1];
    destination->border_color_b = source.borderColor[2];
    destination->border_color_a = source.borderColor[3];
    destination->normalized_coordinates = to_jyppx_bool(source.normalizedCoords != 0);
    destination->max_anisotropy = source.maxAnisotropy;
    destination->mipmap_filter_mode = static_cast<int32_t>(source.mipmapFilterMode);
    destination->mipmap_level_bias = source.mipmapLevelBias;
    destination->min_mipmap_level_clamp = source.minMipmapLevelClamp;
    destination->max_mipmap_level_clamp = source.maxMipmapLevelClamp;
    destination->disable_trilinear_optimization = to_jyppx_bool(source.disableTrilinearOptimization != 0);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    destination->seamless_cubemap = to_jyppx_bool(source.seamlessCubemap != 0);
#else
    destination->seamless_cubemap = JYPPX_FALSE;
#endif
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080 && CUDART_VERSION < 12000
void fill_texture_descriptor_v2(const JYPPX_CudaTextureDescriptor& source, cudaTextureDesc_v2* destination)
{
    destination->addressMode[0] = static_cast<cudaTextureAddressMode>(source.address_mode_x);
    destination->addressMode[1] = static_cast<cudaTextureAddressMode>(source.address_mode_y);
    destination->addressMode[2] = static_cast<cudaTextureAddressMode>(source.address_mode_z);
    destination->filterMode = static_cast<cudaTextureFilterMode>(source.filter_mode);
    destination->readMode = static_cast<cudaTextureReadMode>(source.read_mode);
    destination->sRGB = source.srgb != JYPPX_FALSE ? 1 : 0;
    destination->borderColor[0] = source.border_color_r;
    destination->borderColor[1] = source.border_color_g;
    destination->borderColor[2] = source.border_color_b;
    destination->borderColor[3] = source.border_color_a;
    destination->normalizedCoords = source.normalized_coordinates != JYPPX_FALSE ? 1 : 0;
    destination->maxAnisotropy = source.max_anisotropy;
    destination->mipmapFilterMode = static_cast<cudaTextureFilterMode>(source.mipmap_filter_mode);
    destination->mipmapLevelBias = source.mipmap_level_bias;
    destination->minMipmapLevelClamp = source.min_mipmap_level_clamp;
    destination->maxMipmapLevelClamp = source.max_mipmap_level_clamp;
    destination->disableTrilinearOptimization = source.disable_trilinear_optimization != JYPPX_FALSE ? 1 : 0;
    destination->seamlessCubemap = source.seamless_cubemap != JYPPX_FALSE ? 1 : 0;
}

void copy_texture_descriptor_v2(const cudaTextureDesc_v2& source, JYPPX_CudaTextureDescriptor* destination)
{
    destination->address_mode_x = static_cast<int32_t>(source.addressMode[0]);
    destination->address_mode_y = static_cast<int32_t>(source.addressMode[1]);
    destination->address_mode_z = static_cast<int32_t>(source.addressMode[2]);
    destination->filter_mode = static_cast<int32_t>(source.filterMode);
    destination->read_mode = static_cast<int32_t>(source.readMode);
    destination->srgb = to_jyppx_bool(source.sRGB != 0);
    destination->border_color_r = source.borderColor[0];
    destination->border_color_g = source.borderColor[1];
    destination->border_color_b = source.borderColor[2];
    destination->border_color_a = source.borderColor[3];
    destination->normalized_coordinates = to_jyppx_bool(source.normalizedCoords != 0);
    destination->max_anisotropy = source.maxAnisotropy;
    destination->mipmap_filter_mode = static_cast<int32_t>(source.mipmapFilterMode);
    destination->mipmap_level_bias = source.mipmapLevelBias;
    destination->min_mipmap_level_clamp = source.minMipmapLevelClamp;
    destination->max_mipmap_level_clamp = source.maxMipmapLevelClamp;
    destination->disable_trilinear_optimization = to_jyppx_bool(source.disableTrilinearOptimization != 0);
    destination->seamless_cubemap = to_jyppx_bool(source.seamlessCubemap != 0);
}
#endif

cudaResourceDesc make_array_resource_descriptor(const jyppx::cuda::ArrayObject* array)
{
    cudaResourceDesc descriptor{};
    descriptor.resType = cudaResourceTypeArray;
    descriptor.res.array.array = array->handle;
    return descriptor;
}

void copy_resource_descriptor(const cudaResourceDesc& source, JYPPX_CudaResourceDescriptorSnapshot* destination)
{
    destination->resource_type = static_cast<int32_t>(source.resType);
    switch (source.resType)
    {
    case cudaResourceTypeArray:
        destination->has_array = to_jyppx_bool(source.res.array.array != nullptr);
        break;
    case cudaResourceTypeMipmappedArray:
        destination->has_mipmapped_array = to_jyppx_bool(source.res.mipmap.mipmap != nullptr);
        break;
    case cudaResourceTypeLinear:
        destination->has_device_pointer = to_jyppx_bool(source.res.linear.devPtr != nullptr);
        destination->size_in_bytes = static_cast<uint64_t>(source.res.linear.sizeInBytes);
        break;
    case cudaResourceTypePitch2D:
        destination->has_device_pointer = to_jyppx_bool(source.res.pitch2D.devPtr != nullptr);
        destination->width = static_cast<uint64_t>(source.res.pitch2D.width);
        destination->height = static_cast<uint64_t>(source.res.pitch2D.height);
        destination->pitch_in_bytes = static_cast<uint64_t>(source.res.pitch2D.pitchInBytes);
        break;
    default:
        break;
    }
}

void copy_resource_view_descriptor(const cudaResourceViewDesc& source, JYPPX_CudaTextureResourceViewSnapshot* destination)
{
    destination->is_specified = JYPPX_TRUE;
    destination->format = static_cast<int32_t>(source.format);
    destination->width = static_cast<uint64_t>(source.width);
    destination->height = static_cast<uint64_t>(source.height);
    destination->depth = static_cast<uint64_t>(source.depth);
    destination->first_mipmap_level = source.firstMipmapLevel;
    destination->last_mipmap_level = source.lastMipmapLevel;
    destination->first_layer = source.firstLayer;
    destination->last_layer = source.lastLayer;
}
#endif

JYPPX_StatusCode create_surface_object_impl(JYPPX_CudaArray* array, JYPPX_CudaSurfaceObject** out_surface) noexcept
{
    return run_texture_surface_noexcept("cudaCreateSurfaceObject", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_surface, "out_surface");
        if (status != JYPPX_STATUS_OK) return status;
        *out_surface = nullptr;
        status = jyppx::cuda::validate_array(array, "array");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        auto* object = new (std::nothrow) jyppx::cuda::SurfaceObject{};
        if (object == nullptr) return JYPPX_STATUS_OUT_OF_MEMORY;
        object->base.magic = jyppx::cuda::kObjectMagic;
        object->base.kind = jyppx::cuda::ObjectKind::SurfaceObject;
        object->handle = 0;
        const cudaResourceDesc resource = make_array_resource_descriptor(reinterpret_cast<jyppx::cuda::ArrayObject*>(array));
        status = jyppx::cuda::map_cuda_status(cudaCreateSurfaceObject(&object->handle, &resource), "cudaCreateSurfaceObject");
        if (status != JYPPX_STATUS_OK)
        {
            delete object;
            return status;
        }
        *out_surface = reinterpret_cast<JYPPX_CudaSurfaceObject*>(object);
        return JYPPX_STATUS_OK;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA surface object creation");
#endif
    });
}

JYPPX_StatusCode destroy_surface_object_impl(JYPPX_CudaSurfaceObject* surface) noexcept
{
    return run_texture_surface_noexcept("cudaDestroySurfaceObject", [&]() {
        if (surface == nullptr) return JYPPX_STATUS_OK;
        auto status = jyppx::cuda::validate_surface_object(surface, "surface");
        if (status != JYPPX_STATUS_OK) return status;
        auto* object = reinterpret_cast<jyppx::cuda::SurfaceObject*>(surface);
#if JYPPX_HAS_CUDA_TOOLKIT
        status = jyppx::cuda::map_cuda_status(cudaDestroySurfaceObject(object->handle), "cudaDestroySurfaceObject");
#else
        status = JYPPX_STATUS_OK;
#endif
        delete object;
        return status;
    });
}

JYPPX_StatusCode get_surface_resource_impl(JYPPX_CudaSurfaceObject* surface, JYPPX_CudaResourceDescriptorSnapshot* out_snapshot) noexcept
{
    return run_texture_surface_noexcept("cudaGetSurfaceObjectResourceDesc", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
        if (status != JYPPX_STATUS_OK) return status;
        clear_resource_descriptor_snapshot(out_snapshot);
        status = jyppx::cuda::validate_surface_object(surface, "surface");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        cudaResourceDesc resource{};
        status = jyppx::cuda::map_cuda_status(
            cudaGetSurfaceObjectResourceDesc(&resource, reinterpret_cast<jyppx::cuda::SurfaceObject*>(surface)->handle),
            "cudaGetSurfaceObjectResourceDesc");
        if (status == JYPPX_STATUS_OK) copy_resource_descriptor(resource, out_snapshot);
        return status;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA surface object resource query");
#endif
    });
}

JYPPX_StatusCode create_texture_object_impl(
    JYPPX_CudaArray* array,
    const JYPPX_CudaTextureDescriptor* descriptor,
    JYPPX_CudaTextureObject** out_texture,
    const bool use_v2) noexcept
{
    const char* operation = use_v2 ? "cudaCreateTextureObject_v2" : "cudaCreateTextureObject";
    return run_texture_surface_noexcept(operation, [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_texture, "out_texture");
        if (status != JYPPX_STATUS_OK) return status;
        *out_texture = nullptr;
        status = jyppx::cuda::validate_array(array, "array");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        const bool supports_seamless = use_v2
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            || true
#endif
            ;
        status = validate_texture_descriptor(descriptor, supports_seamless, operation);
        if (status != JYPPX_STATUS_OK) return status;
        auto* object = new (std::nothrow) jyppx::cuda::TextureObject{};
        if (object == nullptr) return JYPPX_STATUS_OUT_OF_MEMORY;
        object->base.magic = jyppx::cuda::kObjectMagic;
        object->base.kind = jyppx::cuda::ObjectKind::TextureObject;
        object->has_resource_view = false;
        object->handle = 0;
        const cudaResourceDesc resource = make_array_resource_descriptor(reinterpret_cast<jyppx::cuda::ArrayObject*>(array));
        if (use_v2)
        {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080 && CUDART_VERSION < 12000
            cudaTextureDesc_v2 native_descriptor{};
            fill_texture_descriptor_v2(*descriptor, &native_descriptor);
            status = jyppx::cuda::map_cuda_status(
                cudaCreateTextureObject_v2(&object->handle, &resource, &native_descriptor, nullptr),
                "cudaCreateTextureObject_v2");
#else
            delete object;
            jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "cudaCreateTextureObject_v2 is only exposed by the CUDA 11.8 descriptor ABI.");
            return JYPPX_STATUS_NOT_SUPPORTED;
#endif
        }
        else
        {
            cudaTextureDesc native_descriptor{};
            fill_texture_descriptor_common(*descriptor, &native_descriptor);
            status = jyppx::cuda::map_cuda_status(
                cudaCreateTextureObject(&object->handle, &resource, &native_descriptor, nullptr),
                "cudaCreateTextureObject");
        }
        if (status != JYPPX_STATUS_OK)
        {
            delete object;
            return status;
        }
        *out_texture = reinterpret_cast<JYPPX_CudaTextureObject*>(object);
        return JYPPX_STATUS_OK;
#else
        (void)descriptor;
        (void)use_v2;
        return jyppx::cuda::report_cuda_dependency_missing("CUDA texture object creation");
#endif
    });
}

JYPPX_StatusCode destroy_texture_object_impl(JYPPX_CudaTextureObject* texture) noexcept
{
    return run_texture_surface_noexcept("cudaDestroyTextureObject", [&]() {
        if (texture == nullptr) return JYPPX_STATUS_OK;
        auto status = jyppx::cuda::validate_texture_object(texture, "texture");
        if (status != JYPPX_STATUS_OK) return status;
        auto* object = reinterpret_cast<jyppx::cuda::TextureObject*>(texture);
#if JYPPX_HAS_CUDA_TOOLKIT
        status = jyppx::cuda::map_cuda_status(cudaDestroyTextureObject(object->handle), "cudaDestroyTextureObject");
#else
        status = JYPPX_STATUS_OK;
#endif
        delete object;
        return status;
    });
}

JYPPX_StatusCode get_texture_resource_impl(JYPPX_CudaTextureObject* texture, JYPPX_CudaResourceDescriptorSnapshot* out_snapshot) noexcept
{
    return run_texture_surface_noexcept("cudaGetTextureObjectResourceDesc", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
        if (status != JYPPX_STATUS_OK) return status;
        clear_resource_descriptor_snapshot(out_snapshot);
        status = jyppx::cuda::validate_texture_object(texture, "texture");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        cudaResourceDesc resource{};
        status = jyppx::cuda::map_cuda_status(
            cudaGetTextureObjectResourceDesc(&resource, reinterpret_cast<jyppx::cuda::TextureObject*>(texture)->handle),
            "cudaGetTextureObjectResourceDesc");
        if (status == JYPPX_STATUS_OK) copy_resource_descriptor(resource, out_snapshot);
        return status;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA texture object resource query");
#endif
    });
}

JYPPX_StatusCode get_texture_descriptor_impl(
    JYPPX_CudaTextureObject* texture,
    JYPPX_CudaTextureDescriptor* out_descriptor,
    const bool use_v2) noexcept
{
    const char* operation = use_v2 ? "cudaGetTextureObjectTextureDesc_v2" : "cudaGetTextureObjectTextureDesc";
    return run_texture_surface_noexcept(operation, [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_descriptor, "out_descriptor");
        if (status != JYPPX_STATUS_OK) return status;
        clear_texture_descriptor(out_descriptor);
        status = jyppx::cuda::validate_texture_object(texture, "texture");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        const auto handle = reinterpret_cast<jyppx::cuda::TextureObject*>(texture)->handle;
        if (use_v2)
        {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080 && CUDART_VERSION < 12000
            cudaTextureDesc_v2 descriptor{};
            status = jyppx::cuda::map_cuda_status(cudaGetTextureObjectTextureDesc_v2(&descriptor, handle), operation);
            if (status == JYPPX_STATUS_OK) copy_texture_descriptor_v2(descriptor, out_descriptor);
            return status;
#else
            jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "cudaGetTextureObjectTextureDesc_v2 is only exposed by the CUDA 11.8 descriptor ABI.");
            return JYPPX_STATUS_NOT_SUPPORTED;
#endif
        }
        cudaTextureDesc descriptor{};
        status = jyppx::cuda::map_cuda_status(cudaGetTextureObjectTextureDesc(&descriptor, handle), operation);
        if (status == JYPPX_STATUS_OK) copy_texture_descriptor_common(descriptor, out_descriptor);
        return status;
#else
        (void)use_v2;
        return jyppx::cuda::report_cuda_dependency_missing("CUDA texture descriptor query");
#endif
    });
}

JYPPX_StatusCode get_texture_resource_view_impl(JYPPX_CudaTextureObject* texture, JYPPX_CudaTextureResourceViewSnapshot* out_snapshot) noexcept
{
    return run_texture_surface_noexcept("cudaGetTextureObjectResourceViewDesc", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
        if (status != JYPPX_STATUS_OK) return status;
        clear_texture_resource_view_snapshot(out_snapshot);
        status = jyppx::cuda::validate_texture_object(texture, "texture");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT
        auto* object = reinterpret_cast<jyppx::cuda::TextureObject*>(texture);
        cudaResourceViewDesc descriptor{};
        status = jyppx::cuda::map_cuda_status(
            cudaGetTextureObjectResourceViewDesc(&descriptor, object->handle),
            "cudaGetTextureObjectResourceViewDesc");
        // Some CUDA versions leave the output undefined when creation used a null view descriptor.
        if (status == JYPPX_STATUS_OK && object->has_resource_view)
        {
            copy_resource_view_descriptor(descriptor, out_snapshot);
        }
        return status;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA texture resource view query");
#endif
    });
}
}

#if defined(_MSC_VER) && defined(_WIN32)
#define JYPPX_CUDA_TEXTURE_SURFACE_GUARD(operation, call) \
    __try { return call; } \
    __except (1) { return report_texture_surface_exception(operation, "windows-seh-caught", "A Windows structured exception was contained by the CUDA texture/surface bridge."); }
#else
#define JYPPX_CUDA_TEXTURE_SURFACE_GUARD(operation, call) return call
#endif

JYPPX_StatusCode jyppx_cuda_surface_object_create_array_owner_safe(JYPPX_CudaArray* array, JYPPX_CudaSurfaceObject** out_surface)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaCreateSurfaceObject", create_surface_object_impl(array, out_surface));
}

JYPPX_StatusCode jyppx_cuda_surface_object_destroy_safe(JYPPX_CudaSurfaceObject* surface)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaDestroySurfaceObject", destroy_surface_object_impl(surface));
}

JYPPX_StatusCode jyppx_cuda_surface_object_get_resource_snapshot_safe(JYPPX_CudaSurfaceObject* surface, JYPPX_CudaResourceDescriptorSnapshot* out_snapshot)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaGetSurfaceObjectResourceDesc", get_surface_resource_impl(surface, out_snapshot));
}

JYPPX_StatusCode jyppx_cuda_texture_object_create_array_owner_safe(JYPPX_CudaArray* array, const JYPPX_CudaTextureDescriptor* descriptor, JYPPX_CudaTextureObject** out_texture)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaCreateTextureObject", create_texture_object_impl(array, descriptor, out_texture, false));
}

JYPPX_StatusCode jyppx_cuda_texture_object_create_array_owner_v2_safe(JYPPX_CudaArray* array, const JYPPX_CudaTextureDescriptor* descriptor, JYPPX_CudaTextureObject** out_texture)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaCreateTextureObject_v2", create_texture_object_impl(array, descriptor, out_texture, true));
}

JYPPX_StatusCode jyppx_cuda_texture_object_destroy_safe(JYPPX_CudaTextureObject* texture)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaDestroyTextureObject", destroy_texture_object_impl(texture));
}

JYPPX_StatusCode jyppx_cuda_texture_object_get_resource_snapshot_safe(JYPPX_CudaTextureObject* texture, JYPPX_CudaResourceDescriptorSnapshot* out_snapshot)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaGetTextureObjectResourceDesc", get_texture_resource_impl(texture, out_snapshot));
}

JYPPX_StatusCode jyppx_cuda_texture_object_get_descriptor_safe(JYPPX_CudaTextureObject* texture, JYPPX_CudaTextureDescriptor* out_descriptor)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaGetTextureObjectTextureDesc", get_texture_descriptor_impl(texture, out_descriptor, false));
}

JYPPX_StatusCode jyppx_cuda_texture_object_get_descriptor_v2_safe(JYPPX_CudaTextureObject* texture, JYPPX_CudaTextureDescriptor* out_descriptor)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaGetTextureObjectTextureDesc_v2", get_texture_descriptor_impl(texture, out_descriptor, true));
}

JYPPX_StatusCode jyppx_cuda_texture_object_get_resource_view_snapshot_safe(JYPPX_CudaTextureObject* texture, JYPPX_CudaTextureResourceViewSnapshot* out_snapshot)
{
    JYPPX_CUDA_TEXTURE_SURFACE_GUARD("cudaGetTextureObjectResourceViewDesc", get_texture_resource_view_impl(texture, out_snapshot));
}

#undef JYPPX_CUDA_TEXTURE_SURFACE_GUARD
