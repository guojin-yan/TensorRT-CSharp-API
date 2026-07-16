namespace
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
struct Trt8ShapeBindingSetState
{
    nvinfer1::IExecutionContext* context;
    int32_t binding_index;
    const int32_t* values;
    int32_t value_count;
};

static JYPPX_StatusCode trt8_shape_binding_set_body(void* raw_state)
{
    auto* state = static_cast<Trt8ShapeBindingSetState*>(raw_state);
    const nvinfer1::ICudaEngine& engine = state->context->getEngine();
    const int32_t binding_count = engine.getNbBindings();
    if (state->binding_index < 0 || state->binding_index >= binding_count)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Shape binding index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (!engine.bindingIsInput(state->binding_index) || !engine.isShapeBinding(state->binding_index))
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Binding must be an input shape binding.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const nvinfer1::Dims dimensions = engine.getBindingDimensions(state->binding_index);
    if (dimensions.nbDims <= 0 || dimensions.nbDims > 8)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Shape binding dimensions are invalid.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    int64_t required_count = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index)
    {
        const int32_t extent = dimensions.d[index];
        if (extent <= 0 || required_count > 1000000 / extent)
        {
            jyppx::common::set_last_error(
                JYPPX_ERROR_CATEGORY_TENSORRT,
                "Shape binding value count is outside the supported range.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        required_count *= extent;
    }
    if (required_count != state->value_count)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Shape binding value count does not match the binding dimensions.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return state->context->setInputShapeBinding(state->binding_index, state->values)
        ? JYPPX_STATUS_OK
        : JYPPX_STATUS_RUNTIME_ERROR;
}

static JYPPX_StatusCode trt8_shape_binding_set_with_seh_guard(
    Trt8ShapeBindingSetState* state,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return trt8_shape_binding_set_body(state);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return trt8_shape_binding_set_body(state);
#endif
}

static JYPPX_StatusCode run_trt8_shape_binding_set(
    Trt8ShapeBindingSetState* state,
    const char* feature_name)
{
    try
    {
        return trt8_shape_binding_set_with_seh_guard(state, feature_name);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_input_shape_binding(
    JYPPX_TensorRtExecutionContext* context,
    int32_t binding_index,
    const int32_t* values,
    int32_t value_count,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (binding_index < 0 || values == nullptr || value_count <= 0 || value_count > 1000000)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Shape binding input requires a non-negative index and 1 to 1000000 copied values.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(
        context,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT,
        "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    Trt8ShapeBindingSetState state{payload, binding_index, values, value_count};
    status = run_trt8_shape_binding_set(&state, "execution context input shape binding set");
    if (status == JYPPX_STATUS_RUNTIME_ERROR)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "IExecutionContext::setInputShapeBinding returned false.");
    }
    if (status == JYPPX_STATUS_OK) { *out_set = JYPPX_TRUE; }
    return status;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "execution context input shape binding set");
#else
    return jyppx::tensorrt::report_vendor_missing(
        kLine,
        "execution context input shape binding set");
#endif
}
