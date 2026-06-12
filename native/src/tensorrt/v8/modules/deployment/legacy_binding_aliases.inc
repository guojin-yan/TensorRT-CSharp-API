#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
JYPPX_StatusCode get_trt8_legacy_cuda_engine_payload(JYPPX_TensorRtCudaEngine* engine, nvinfer1::ICudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_engine = nullptr;
    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_engine = engine_payload;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_trt8_legacy_binding_index(nvinfer1::ICudaEngine* engine_payload, int32_t binding_index)
{
    if (binding_index < 0 || binding_index >= engine_payload->getNbBindings())
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 binding index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode report_trt8_legacy_binding_unavailable(const char* feature_name)
{
#if JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_nb_bindings(JYPPX_TensorRtCudaEngine* engine, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = engine_payload->getNbBindings();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getNbBindings");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_name(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_trt8_legacy_binding_index(engine_payload, binding_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getBindingName(binding_index), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine; (void)binding_index; (void)output_buffer; (void)output_buffer_size;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingName");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_index(JYPPX_TensorRtCudaEngine* engine, const char* binding_name, int32_t* out_index)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_index, "out_index");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_index = -1;
    status = validate_c_string(binding_name, "binding_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_index = engine_payload->getBindingIndex(binding_name);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingIndex");
#endif
}

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
template <typename TGetter>
JYPPX_StatusCode get_trt8_legacy_binding_int(
    JYPPX_TensorRtCudaEngine* engine,
    int32_t binding_index,
    int32_t* output,
    const char* output_name,
    TGetter getter)
{
    auto status = jyppx::tensorrt::validate_output_pointer(output, output_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *output = 0;
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_trt8_legacy_binding_index(engine_payload, binding_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    *output = static_cast<int32_t>(getter(engine_payload, binding_index));
    return JYPPX_STATUS_OK;
}

template <typename TGetter>
JYPPX_StatusCode get_trt8_legacy_binding_bool(
    JYPPX_TensorRtCudaEngine* engine,
    int32_t binding_index,
    JYPPX_Boolean* output,
    const char* output_name,
    TGetter getter)
{
    auto status = jyppx::tensorrt::validate_output_pointer(output, output_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *output = JYPPX_FALSE;
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_trt8_legacy_binding_index(engine_payload, binding_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    *output = getter(engine_payload, binding_index) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt8_cuda_engine_binding_is_input(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, JYPPX_Boolean* out_is_input)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_bool(engine, binding_index, out_is_input, "out_is_input", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->bindingIsInput(index); });
#else
    (void)engine; (void)binding_index; (void)out_is_input;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::bindingIsInput");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_data_type(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t* out_data_type)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_int(engine, binding_index, out_data_type, "out_data_type", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->getBindingDataType(index); });
#else
    (void)engine; (void)binding_index; (void)out_data_type;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingDataType");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_bytes_per_component(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t* out_bytes)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_int(engine, binding_index, out_bytes, "out_bytes", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->getBindingBytesPerComponent(index); });
#else
    (void)engine; (void)binding_index; (void)out_bytes;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingBytesPerComponent");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_components_per_element(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t* out_components)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_int(engine, binding_index, out_components, "out_components", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->getBindingComponentsPerElement(index); });
#else
    (void)engine; (void)binding_index; (void)out_components;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingComponentsPerElement");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_format(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t* out_format)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_int(engine, binding_index, out_format, "out_format", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->getBindingFormat(index); });
#else
    (void)engine; (void)binding_index; (void)out_format;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingFormat");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_vectorized_dim(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, int32_t* out_dim)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_int(engine, binding_index, out_dim, "out_dim", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->getBindingVectorizedDim(index); });
#else
    (void)engine; (void)binding_index; (void)out_dim;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingVectorizedDim");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_is_execution_binding(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, JYPPX_Boolean* out_is_execution_binding)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_bool(engine, binding_index, out_is_execution_binding, "out_is_execution_binding", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->isExecutionBinding(index); });
#else
    (void)engine; (void)binding_index; (void)out_is_execution_binding;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::isExecutionBinding");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_is_shape_binding(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, JYPPX_Boolean* out_is_shape_binding)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return get_trt8_legacy_binding_bool(engine, binding_index, out_is_shape_binding, "out_is_shape_binding", [](nvinfer1::ICudaEngine* payload, int32_t index) { return payload->isShapeBinding(index); });
#else
    (void)engine; (void)binding_index; (void)out_is_shape_binding;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::isShapeBinding");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_dimensions(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, JYPPX_TensorRtDims* out_dimensions)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dimensions, "out_dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_dimensions, 0, sizeof(*out_dimensions));
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_trt8_legacy_binding_index(engine_payload, binding_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(engine_payload->getBindingDimensions(binding_index), out_dimensions);
    return JYPPX_STATUS_OK;
#else
    (void)engine; (void)binding_index;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingDimensions");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_get_binding_format_desc(JYPPX_TensorRtCudaEngine* engine, int32_t binding_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_trt8_legacy_binding_index(engine_payload, binding_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getBindingFormatDesc(binding_index), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine; (void)binding_index; (void)output_buffer; (void)output_buffer_size;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::getBindingFormatDesc");
#endif
}

JYPPX_StatusCode jyppx_trt8_cuda_engine_has_implicit_batch_dimension(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_implicit_batch)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_implicit_batch, "out_has_implicit_batch");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_implicit_batch = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_trt8_legacy_cuda_engine_payload(engine, &engine_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_implicit_batch = engine_payload->hasImplicitBatchDimension() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return report_trt8_legacy_binding_unavailable("ICudaEngine::hasImplicitBatchDimension");
#endif
}
