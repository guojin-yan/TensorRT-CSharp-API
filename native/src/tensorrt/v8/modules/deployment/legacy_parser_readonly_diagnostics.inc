namespace
{
constexpr const char* kLegacyUffVersionFeature = "TRT8 legacy UFF required-version snapshot";
constexpr const char* kLegacyCaffeBinaryProtoFeature = "TRT8 legacy Caffe binaryproto copied snapshot";

void reset_legacy_uff_version_outputs(
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_patch)
{
    *out_major = 0;
    *out_minor = 0;
    *out_patch = 0;
}

void reset_legacy_binary_proto_outputs(
    size_t* out_required_size,
    JYPPX_TensorRtDims* out_dimensions,
    int32_t* out_data_type)
{
    *out_required_size = 0;
    out_dimensions->nb_dims = 0;
    for (int32_t index = 0; index < 8; ++index)
    {
        out_dimensions->d[index] = 0;
    }
    *out_data_type = 0;
}

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_LEGACY_PARSERS && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
JYPPX_StatusCode get_legacy_uff_required_version_unprotected(
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_patch)
{
    std::unique_ptr<nvuffparser::IUffParser> parser(nvuffparser::createUffParser());
    if (!parser)
    {
        return report_null_vendor_object("nvuffparser::createUffParser");
    }

    *out_major = parser->getUffRequiredVersionMajor();
    *out_minor = parser->getUffRequiredVersionMinor();
    *out_patch = parser->getUffRequiredVersionPatch();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode copy_legacy_caffe_binary_proto_unprotected(
    const char* file_path,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    JYPPX_TensorRtDims* out_dimensions,
    int32_t* out_data_type)
{
    std::unique_ptr<nvcaffeparser1::ICaffeParser> parser(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return report_null_vendor_object("nvcaffeparser1::createCaffeParser");
    }

    std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob> blob(parser->parseBinaryProto(file_path));
    if (!blob)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ICaffeParser::parseBinaryProto returned null.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    const nvinfer1::Dims4 dimensions = blob->getDimensions();
    if (dimensions.nbDims < 0 || dimensions.nbDims > 8)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto dimensions exceed the bridge ABI rank limit.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    size_t element_count = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index)
    {
        const int32_t extent = dimensions.d[index];
        if (extent < 0)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto dimensions contain a negative extent.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        if (extent != 0 && element_count > SIZE_MAX / static_cast<size_t>(extent))
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto element count exceeds the native size limit.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        element_count *= static_cast<size_t>(extent);
    }

    const int32_t data_type = static_cast<int32_t>(blob->getDataType());
    const size_t element_size = get_data_type_size(data_type);
    if (element_size == 0 || element_count > SIZE_MAX / element_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto uses an unsupported data type or exceeds the native byte-size limit.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t required_size = element_count * element_size;
    out_dimensions->nb_dims = dimensions.nbDims;
    for (int32_t index = 0; index < 8; ++index)
    {
        out_dimensions->d[index] = index < dimensions.nbDims ? dimensions.d[index] : 0;
    }
    *out_data_type = data_type;
    *out_required_size = required_size;

    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }
    if (output_buffer_size < required_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    const void* data = blob->getData();
    if (required_size > 0 && data == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IBinaryProtoBlob::getData returned null for non-empty data.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    if (required_size > 0)
    {
        std::memcpy(output_buffer, data, required_size);
    }
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_legacy_uff_required_version_with_seh_guard(
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_patch)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return get_legacy_uff_required_version_unprotected(out_major, out_minor, out_patch);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_legacy_uff_version_outputs(out_major, out_minor, out_patch);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kLegacyUffVersionFeature, seh_exception_code);
    }
#else
    return get_legacy_uff_required_version_unprotected(out_major, out_minor, out_patch);
#endif
}

JYPPX_StatusCode copy_legacy_caffe_binary_proto_with_seh_guard(
    const char* file_path,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    JYPPX_TensorRtDims* out_dimensions,
    int32_t* out_data_type)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return copy_legacy_caffe_binary_proto_unprotected(
            file_path, output_buffer, output_buffer_size, out_required_size, out_dimensions, out_data_type);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_legacy_binary_proto_outputs(out_required_size, out_dimensions, out_data_type);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kLegacyCaffeBinaryProtoFeature, seh_exception_code);
    }
#else
    return copy_legacy_caffe_binary_proto_unprotected(
        file_path, output_buffer, output_buffer_size, out_required_size, out_dimensions, out_data_type);
#endif
}
#endif
}

JYPPX_StatusCode jyppx_trt8_legacy_uff_get_required_version(
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_patch)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_patch, "out_patch");
    if (status != JYPPX_STATUS_OK) { return status; }
    reset_legacy_uff_version_outputs(out_major, out_minor, out_patch);

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_LEGACY_PARSERS && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    try
    {
        return get_legacy_uff_required_version_with_seh_guard(out_major, out_minor, out_patch);
    }
    catch (const std::exception& exception)
    {
        reset_legacy_uff_version_outputs(out_major, out_minor, out_patch);
        return jyppx::tensorrt::report_vendor_exception(kLine, kLegacyUffVersionFeature, exception.what());
    }
    catch (...)
    {
        reset_legacy_uff_version_outputs(out_major, out_minor, out_patch);
        return jyppx::tensorrt::report_vendor_exception(kLine, kLegacyUffVersionFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 legacy parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kLegacyUffVersionFeature);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, kLegacyUffVersionFeature);
#endif
}

JYPPX_StatusCode jyppx_trt8_legacy_caffe_binary_proto_copy(
    const char* file_path,
    uint8_t* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    JYPPX_TensorRtDims* out_dimensions,
    int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_dimensions, "out_dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    reset_legacy_binary_proto_outputs(out_required_size, out_dimensions, out_data_type);

    status = validate_c_string(file_path, "file_path");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (output_buffer == nullptr && output_buffer_size > 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Caffe binaryproto output buffer must not be null when its size is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_LEGACY_PARSERS && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    try
    {
        return copy_legacy_caffe_binary_proto_with_seh_guard(
            file_path, output_buffer, output_buffer_size, out_required_size, out_dimensions, out_data_type);
    }
    catch (const std::exception& exception)
    {
        reset_legacy_binary_proto_outputs(out_required_size, out_dimensions, out_data_type);
        return jyppx::tensorrt::report_vendor_exception(kLine, kLegacyCaffeBinaryProtoFeature, exception.what());
    }
    catch (...)
    {
        reset_legacy_binary_proto_outputs(out_required_size, out_dimensions, out_data_type);
        return jyppx::tensorrt::report_vendor_exception(kLine, kLegacyCaffeBinaryProtoFeature, "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 legacy parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#elif JYPPX_HAS_TENSORRT
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, kLegacyCaffeBinaryProtoFeature);
#else
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, kLegacyCaffeBinaryProtoFeature);
#endif
}
