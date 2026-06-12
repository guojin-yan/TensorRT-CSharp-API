template <typename TLayer>
static JYPPX_StatusCode get_typed_layer_for_api(
    JYPPX_TensorRtLayer* layer,
    const nvinfer1::LayerType expected_type,
    const char* type_name,
    const char* feature_name,
    TLayer** out_layer)
{
    *out_layer = nullptr;
    nvinfer1::ILayer* base_layer = nullptr;
    auto status = get_layer_payload_for_api(layer, &base_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (base_layer->getType() != expected_type)
    {
        std::ostringstream builder;
        builder << "Layer handle is not a TensorRT " << type_name << " layer.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_layer = static_cast<TLayer*>(base_layer);
    return JYPPX_STATUS_OK;
}

#define JYPPX_TRT11_LAYER_INT_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t* out_value) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_value = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_value = static_cast<int32_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_INT_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, VALUE_TYPE) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t value) \
    { \
        if (value < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<VALUE_TYPE>(value)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_UINT_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, uint32_t* out_value) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_value = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_value = static_cast<uint32_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_UINT_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, uint32_t value) \
    { \
        if (value == 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(value); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_DIMS_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* PARAM_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(PARAM_NAME, 0, sizeof(*PARAM_NAME)); \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        copy_dims(typed_layer->METHOD(), PARAM_NAME); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_DIMS_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* PARAM_NAME) \
    { \
        nvinfer1::Dims native_dims{}; \
        auto status = make_dims(PARAM_NAME, &native_dims, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(native_dims); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_DOUBLE_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, double* PARAM_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = 0.0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = static_cast<double>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_DOUBLE_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, double PARAM_NAME) \
    { \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<float>(PARAM_NAME)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_BOOL_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* PARAM_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = JYPPX_FALSE; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = typed_layer->METHOD() ? JYPPX_TRUE : JYPPX_FALSE; \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_BOOL_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_Boolean PARAM_NAME) \
    { \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME != JYPPX_FALSE); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_INT64_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int64_t* PARAM_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *PARAM_NAME = static_cast<int64_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_INT64_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int64_t PARAM_NAME) \
    { \
        if (PARAM_NAME < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME); \
        return JYPPX_STATUS_OK; \
    }

JYPPX_StatusCode jyppx_trt11_matrix_multiply_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t operation)
{
    if (input_index < 0 || input_index > 1 || operation < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IMatrixMultiplyLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMATRIX_MULTIPLY, "matrix multiply", "matrix multiply layer operation set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setOperation(input_index, static_cast<nvinfer1::MatrixOperation>(operation));
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_matrix_multiply_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t input_index, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (input_index < 0 || input_index > 1) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_operation = 0;
    nvinfer1::IMatrixMultiplyLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMATRIX_MULTIPLY, "matrix multiply", "matrix multiply layer operation query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_operation = static_cast<int32_t>(typed_layer->getOperation(input_index));
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_shuffle_layer_set_reshape_dimensions, nvinfer1::IShuffleLayer, nvinfer1::LayerType::kSHUFFLE, "shuffle", setReshapeDimensions, dims)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_shuffle_layer_get_reshape_dimensions, nvinfer1::IShuffleLayer, nvinfer1::LayerType::kSHUFFLE, "shuffle", getReshapeDimensions, out_dims)
JYPPX_StatusCode jyppx_trt11_shuffle_layer_set_first_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation)
{
    nvinfer1::Permutation native_permutation{};
    auto status = make_permutation(permutation, &native_permutation, "permutation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IShuffleLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kSHUFFLE, "shuffle", "shuffle layer first transpose set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setFirstTranspose(native_permutation);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_shuffle_layer_get_first_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_permutation, "out_permutation");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_permutation, 0, sizeof(*out_permutation));
    nvinfer1::IShuffleLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kSHUFFLE, "shuffle", "shuffle layer first transpose query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_permutation(typed_layer->getFirstTranspose(), get_layer_input_rank(typed_layer), out_permutation);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_shuffle_layer_set_second_transpose(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* permutation)
{
    nvinfer1::Permutation native_permutation{};
    auto status = make_permutation(permutation, &native_permutation, "permutation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IShuffleLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kSHUFFLE, "shuffle", "shuffle layer second transpose set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSecondTranspose(native_permutation);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_shuffle_layer_get_second_transpose(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_permutation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_permutation, "out_permutation");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_permutation, 0, sizeof(*out_permutation));
    nvinfer1::IShuffleLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kSHUFFLE, "shuffle", "shuffle layer second transpose query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_permutation(typed_layer->getSecondTranspose(), get_shuffle_second_transpose_rank(typed_layer), out_permutation);
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_shuffle_layer_set_zero_is_placeholder, nvinfer1::IShuffleLayer, nvinfer1::LayerType::kSHUFFLE, "shuffle", setZeroIsPlaceholder, zero_is_placeholder)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_shuffle_layer_get_zero_is_placeholder, nvinfer1::IShuffleLayer, nvinfer1::LayerType::kSHUFFLE, "shuffle", getZeroIsPlaceholder, out_zero_is_placeholder)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_reduce_layer_set_operation, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", setOperation, nvinfer1::ReduceOperation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_reduce_layer_get_operation, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", getOperation)
JYPPX_TRT11_LAYER_UINT_SET(jyppx_trt11_reduce_layer_set_axes, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", setReduceAxes)
JYPPX_TRT11_LAYER_UINT_GET(jyppx_trt11_reduce_layer_get_axes, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", getReduceAxes)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_reduce_layer_set_keep_dimensions, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", setKeepDimensions, keep_dimensions)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_reduce_layer_get_keep_dimensions, nvinfer1::IReduceLayer, nvinfer1::LayerType::kREDUCE, "reduce", getKeepDimensions)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_elementwise_layer_set_operation, nvinfer1::IElementWiseLayer, nvinfer1::LayerType::kELEMENTWISE, "elementwise", setOperation, nvinfer1::ElementWiseOperation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_elementwise_layer_get_operation, nvinfer1::IElementWiseLayer, nvinfer1::LayerType::kELEMENTWISE, "elementwise", getOperation)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_concatenation_layer_set_axis, nvinfer1::IConcatenationLayer, nvinfer1::LayerType::kCONCATENATION, "concatenation", setAxis, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_concatenation_layer_get_axis, nvinfer1::IConcatenationLayer, nvinfer1::LayerType::kCONCATENATION, "concatenation", getAxis)

JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_slice_layer_set_start, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", setStart, start)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_slice_layer_get_start, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getStart, out_start)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_slice_layer_set_size, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", setSize, size)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_slice_layer_get_size, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getSize, out_size)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_slice_layer_set_stride, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", setStride, stride)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_slice_layer_get_stride, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getStride, out_stride)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_slice_layer_set_mode, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", setMode, nvinfer1::SampleMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_slice_layer_get_mode, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getMode)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_slice_layer_set_axes, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", setAxes, axes)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_slice_layer_get_axes, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getAxes, out_axes)

JYPPX_TRT11_LAYER_UINT_SET(jyppx_trt11_softmax_layer_set_axes, nvinfer1::ISoftMaxLayer, nvinfer1::LayerType::kSOFTMAX, "softmax", setAxes)
JYPPX_TRT11_LAYER_UINT_GET(jyppx_trt11_softmax_layer_get_axes, nvinfer1::ISoftMaxLayer, nvinfer1::LayerType::kSOFTMAX, "softmax", getAxes)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_unary_layer_set_operation, nvinfer1::IUnaryLayer, nvinfer1::LayerType::kUNARY, "unary", setOperation, nvinfer1::UnaryOperation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_unary_layer_get_operation, nvinfer1::IUnaryLayer, nvinfer1::LayerType::kUNARY, "unary", getOperation)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_topk_layer_set_operation, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", setOperation, nvinfer1::TopKOperation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_topk_layer_get_operation, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", getOperation)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_topk_layer_set_k, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", setK, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_topk_layer_get_k, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", getK)
JYPPX_TRT11_LAYER_UINT_SET(jyppx_trt11_topk_layer_set_axes, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", setReduceAxes)
JYPPX_TRT11_LAYER_UINT_GET(jyppx_trt11_topk_layer_get_axes, nvinfer1::ITopKLayer, nvinfer1::LayerType::kTOPK, "topk", getReduceAxes)

JYPPX_StatusCode jyppx_trt11_topk_layer_set_indices_type(JYPPX_TensorRtLayer* layer, int32_t data_type, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    nvinfer1::ITopKLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kTOPK, "topk", "topk layer indices type set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setIndicesType(static_cast<nvinfer1::DataType>(data_type)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_topk_layer_get_indices_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
    nvinfer1::ITopKLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kTOPK, "topk", "topk layer indices type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(typed_layer->getIndicesType());
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_gather_layer_set_axis, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", setGatherAxis, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_gather_layer_get_axis, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", getGatherAxis)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_gather_layer_set_mode, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", setMode, nvinfer1::GatherMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_gather_layer_get_mode, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", getMode)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_gather_layer_set_nb_elementwise_dims, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", setNbElementWiseDims, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_gather_layer_get_nb_elementwise_dims, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", getNbElementWiseDims)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_scatter_layer_set_mode, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", setMode, nvinfer1::ScatterMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_scatter_layer_get_mode, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", getMode)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_scatter_layer_set_axis, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", setAxis, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_scatter_layer_get_axis, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", getAxis)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_one_hot_layer_set_axis, nvinfer1::IOneHotLayer, nvinfer1::LayerType::kONE_HOT, "one-hot", setAxis, int32_t)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_one_hot_layer_get_axis, nvinfer1::IOneHotLayer, nvinfer1::LayerType::kONE_HOT, "one-hot", getAxis)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_constant_layer_set_dimensions, nvinfer1::IConstantLayer, nvinfer1::LayerType::kCONSTANT, "constant", setDimensions, dimensions)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_constant_layer_get_dimensions, nvinfer1::IConstantLayer, nvinfer1::LayerType::kCONSTANT, "constant", getDimensions, out_dimensions)

JYPPX_StatusCode jyppx_trt11_cumulative_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation)
{
    if (operation < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::ICumulativeLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", "cumulative layer operation set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!typed_layer->setOperation(static_cast<nvinfer1::CumulativeOperation>(operation)))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ICumulativeLayer::setOperation returned false.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_cumulative_layer_get_operation, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getOperation)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_cumulative_layer_set_exclusive, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", setExclusive, exclusive)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_cumulative_layer_get_exclusive, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getExclusive, out_exclusive)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_cumulative_layer_set_reverse, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", setReverse, reverse)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_cumulative_layer_get_reverse, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getReverse, out_reverse)

JYPPX_StatusCode jyppx_trt11_assertion_layer_set_message(JYPPX_TensorRtLayer* layer, const char* message)
{
    auto status = validate_c_string(message, "message");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IAssertionLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kASSERTION, "assertion", "assertion message set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setMessage(message);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_assertion_layer_get_message(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    nvinfer1::IAssertionLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kASSERTION, "assertion", "assertion message query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(typed_layer->getMessage(), output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_einsum_layer_set_equation(JYPPX_TensorRtLayer* layer, const char* equation, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    status = validate_c_string(equation, "equation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IEinsumLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kEINSUM, "einsum", "einsum equation set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setEquation(equation) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_einsum_layer_get_equation(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    nvinfer1::IEinsumLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kEINSUM, "einsum", "einsum equation query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(typed_layer->getEquation(), output_buffer, output_buffer_size, out_required_size);
}

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_grid_sample_layer_set_interpolation_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", setInterpolationMode, nvinfer1::InterpolationMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_grid_sample_layer_get_interpolation_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getInterpolationMode)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_grid_sample_layer_set_align_corners, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", setAlignCorners, align_corners)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_grid_sample_layer_get_align_corners, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getAlignCorners, out_align_corners)

JYPPX_StatusCode jyppx_trt11_grid_sample_layer_set_sample_mode(JYPPX_TensorRtLayer* layer, int32_t sample_mode)
{
    if (sample_mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IGridSampleLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", "grid sample mode set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!typed_layer->setSampleMode(static_cast<nvinfer1::SampleMode>(sample_mode)))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IGridSampleLayer::setSampleMode returned false.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_grid_sample_layer_get_sample_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getSampleMode)

JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_normalization_layer_set_epsilon, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setEpsilon, epsilon)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_normalization_layer_get_epsilon, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getEpsilon, out_epsilon)
JYPPX_TRT11_LAYER_UINT_SET(jyppx_trt11_normalization_layer_set_axes, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setAxes)
JYPPX_TRT11_LAYER_UINT_GET(jyppx_trt11_normalization_layer_get_axes, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getAxes)
JYPPX_TRT11_LAYER_INT64_SET(jyppx_trt11_normalization_layer_set_nb_groups, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setNbGroups, group_count)
JYPPX_TRT11_LAYER_INT64_GET(jyppx_trt11_normalization_layer_get_nb_groups, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getNbGroups, out_group_count)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_normalization_layer_is_v2, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", isV2, out_is_v2)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_dynamic_quantize_layer_set_to_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setToType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_dynamic_quantize_layer_get_to_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getToType)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_dynamic_quantize_layer_set_scale_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setScaleType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_dynamic_quantize_layer_get_scale_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getScaleType)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_dynamic_quantize_layer_set_block_shape, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setBlockShape, block_shape)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_dynamic_quantize_layer_get_block_shape, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getBlockShape, out_block_shape)

JYPPX_StatusCode jyppx_trt11_dynamic_quantize_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    if (axis < -1)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    nvinfer1::IDynamicQuantizeLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", "dynamic quantize axis set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setAxis(axis);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_dynamic_quantize_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = 0;
    nvinfer1::IDynamicQuantizeLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", "dynamic quantize axis query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = typed_layer->getAxis();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_dynamic_quantize_layer_set_block_size(JYPPX_TensorRtLayer* layer, int32_t block_size)
{
    if (block_size <= 0)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    nvinfer1::IDynamicQuantizeLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", "dynamic quantize block size set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBlockSize(block_size);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_dynamic_quantize_layer_get_block_size(JYPPX_TensorRtLayer* layer, int32_t* out_block_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_block_size, "out_block_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_block_size = 0;
    nvinfer1::IDynamicQuantizeLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", "dynamic quantize block size query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_block_size = typed_layer->getBlockSize();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_dequantize_layer_set_block_shape(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* block_shape, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    nvinfer1::Dims native_dims{};
    status = make_dims(block_shape, &native_dims, "block_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IDequantizeLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", "dequantize layer block shape set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setBlockShape(native_dims) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_dequantize_layer_get_block_shape(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* out_block_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_block_shape, "out_block_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_block_shape, 0, sizeof(*out_block_shape));
    nvinfer1::IDequantizeLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", "dequantize layer block shape query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims(typed_layer->getBlockShape(), out_block_shape);
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_dequantize_layer_set_to_type, nvinfer1::IDequantizeLayer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", setToType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_dequantize_layer_get_to_type, nvinfer1::IDequantizeLayer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", getToType)

JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_rotary_embedding_layer_set_interleaved, nvinfer1::IRotaryEmbeddingLayer, nvinfer1::LayerType::kROTARY_EMBEDDING, "rotary embedding", setInterleaved, interleaved)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_rotary_embedding_layer_get_interleaved, nvinfer1::IRotaryEmbeddingLayer, nvinfer1::LayerType::kROTARY_EMBEDDING, "rotary embedding", getInterleaved, out_interleaved)

JYPPX_StatusCode jyppx_trt11_rotary_embedding_layer_set_rotary_embedding_dim(JYPPX_TensorRtLayer* layer, int32_t rotary_embedding_dim, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (rotary_embedding_dim < 0 || (rotary_embedding_dim % 2) != 0)
    {
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    nvinfer1::IRotaryEmbeddingLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kROTARY_EMBEDDING, "rotary embedding", "rotary embedding dimension set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setRotaryEmbeddingDim(rotary_embedding_dim) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_rotary_embedding_layer_get_rotary_embedding_dim, nvinfer1::IRotaryEmbeddingLayer, nvinfer1::LayerType::kROTARY_EMBEDDING, "rotary embedding", getRotaryEmbeddingDim)

JYPPX_StatusCode jyppx_trt11_kv_cache_update_layer_set_cache_mode(JYPPX_TensorRtLayer* layer, int32_t cache_mode, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (cache_mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IKVCacheUpdateLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", "KV cache mode set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setCacheMode(static_cast<nvinfer1::KVCacheMode>(cache_mode)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_kv_cache_update_layer_get_cache_mode, nvinfer1::IKVCacheUpdateLayer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", getCacheMode)

JYPPX_StatusCode jyppx_trt11_kv_cache_update_layer_set_update_form(JYPPX_TensorRtLayer* layer, int32_t update_form, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (update_form < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IKVCacheUpdateLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", "KV cache update form set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setUpdateForm(static_cast<nvinfer1::AttentionIOForm>(update_form)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_kv_cache_update_layer_get_update_form, nvinfer1::IKVCacheUpdateLayer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", getUpdateForm)

JYPPX_StatusCode jyppx_trt11_kv_cache_update_layer_set_update_lengths(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* lengths_tensor, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    nvinfer1::IKVCacheUpdateLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", "KV cache update lengths set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* lengths = nullptr;
    if (lengths_tensor != nullptr)
    {
        status = get_tensor_payload_for_api(lengths_tensor, &lengths, "lengths_tensor");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    *out_success = typed_layer->setUpdateLengths(lengths) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_kv_cache_update_layer_get_update_lengths(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor** out_tensor, JYPPX_Boolean* out_has_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    *out_has_tensor = JYPPX_FALSE;
    nvinfer1::IKVCacheUpdateLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kKVCACHE_UPDATE, "KV cache update", "KV cache update lengths query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* lengths = typed_layer->getUpdateLengths();
    if (lengths == nullptr)
    {
        return JYPPX_STATUS_OK;
    }
    status = create_tensor_reference_handle(lengths, out_tensor);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_tensor = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_activation_layer_set_type, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", setActivationType, nvinfer1::ActivationType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_activation_layer_get_type, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", getActivationType)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_activation_layer_set_alpha, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", setAlpha, alpha)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_activation_layer_get_alpha, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", getAlpha, out_alpha)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_activation_layer_set_beta, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", setBeta, beta)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_activation_layer_get_beta, nvinfer1::IActivationLayer, nvinfer1::LayerType::kACTIVATION, "activation", getBeta, out_beta)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_pooling_layer_set_type, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setPoolingType, nvinfer1::PoolingType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_pooling_layer_get_type, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPoolingType)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_pooling_layer_set_window_size_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setWindowSizeNd, window_size)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_pooling_layer_get_window_size_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getWindowSizeNd, out_window_size)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_pooling_layer_set_stride_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setStrideNd, stride)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_pooling_layer_get_stride_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getStrideNd, out_stride)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_pooling_layer_set_padding_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setPaddingNd, padding)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_pooling_layer_get_padding_nd, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPaddingNd, out_padding)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_pooling_layer_set_blend_factor, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setBlendFactor, blend_factor)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_pooling_layer_get_blend_factor, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getBlendFactor, out_blend_factor)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_pooling_layer_set_average_count_excludes_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setAverageCountExcludesPadding, excludes_padding)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_pooling_layer_get_average_count_excludes_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getAverageCountExcludesPadding, out_excludes_padding)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_pooling_layer_set_pre_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setPrePadding, pre_padding)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_pooling_layer_get_pre_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPrePadding, out_pre_padding)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_pooling_layer_set_post_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setPostPadding, post_padding)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_pooling_layer_get_post_padding, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPostPadding, out_post_padding)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_pooling_layer_set_padding_mode, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", setPaddingMode, nvinfer1::PaddingMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_pooling_layer_get_padding_mode, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPaddingMode)

JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_resize_layer_set_output_dimensions, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setOutputDimensions, dimensions)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_resize_layer_get_output_dimensions, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getOutputDimensions, out_dimensions)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_resize_layer_set_mode, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setResizeMode, nvinfer1::InterpolationMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_resize_layer_get_mode, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getResizeMode)
JYPPX_StatusCode jyppx_trt11_resize_layer_set_resize_mode(JYPPX_TensorRtLayer* layer, int32_t mode)
{
    return jyppx_trt11_resize_layer_set_mode(layer, mode);
}

JYPPX_StatusCode jyppx_trt11_resize_layer_get_resize_mode(JYPPX_TensorRtLayer* layer, int32_t* out_mode)
{
    return jyppx_trt11_resize_layer_get_mode(layer, out_mode);
}

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_resize_layer_set_coordinate_transformation, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setCoordinateTransformation, nvinfer1::ResizeCoordinateTransformation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_resize_layer_get_coordinate_transformation, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getCoordinateTransformation)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_resize_layer_set_selector_for_single_pixel, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setSelectorForSinglePixel, nvinfer1::ResizeSelector)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_resize_layer_get_selector_for_single_pixel, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getSelectorForSinglePixel)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_resize_layer_set_nearest_rounding, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setNearestRounding, nvinfer1::ResizeRoundMode)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_resize_layer_get_nearest_rounding, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getNearestRounding)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_resize_layer_set_cubic_coeff, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setCubicCoeff, cubic_coeff)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_resize_layer_get_cubic_coeff, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getCubicCoeff, out_cubic_coeff)
JYPPX_TRT11_LAYER_BOOL_SET(jyppx_trt11_resize_layer_set_exclude_outside, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", setExcludeOutside, exclude_outside)
JYPPX_TRT11_LAYER_BOOL_GET(jyppx_trt11_resize_layer_get_exclude_outside, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getExcludeOutside, out_exclude_outside)

JYPPX_StatusCode jyppx_trt11_resize_layer_set_scales(JYPPX_TensorRtLayer* layer, const float* scales, int32_t scale_count)
{
    if (scales == nullptr || scale_count <= 0 || scale_count > 8) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IResizeLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kRESIZE, "resize", "resize layer scales set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setScales(scales, scale_count);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_resize_layer_get_scales(JYPPX_TensorRtLayer* layer, float* scales, int32_t scale_capacity, int32_t* out_scale_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_scale_count, "out_scale_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (scale_capacity < 0 || scale_capacity > 8 || (scale_capacity > 0 && scales == nullptr)) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    *out_scale_count = 0;
    nvinfer1::IResizeLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kRESIZE, "resize", "resize layer scales query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_scale_count = typed_layer->getScales(scale_capacity, scales);
    return JYPPX_STATUS_OK;
}

JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_fill_layer_set_dimensions, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setDimensions, dimensions)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_fill_layer_get_dimensions, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getDimensions, out_dimensions)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_fill_layer_set_operation, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setOperation, nvinfer1::FillOperation)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_fill_layer_get_operation, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getOperation)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_fill_layer_set_alpha, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setAlpha, alpha)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_fill_layer_get_alpha, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getAlpha, out_alpha)
JYPPX_TRT11_LAYER_DOUBLE_SET(jyppx_trt11_fill_layer_set_beta, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setBeta, beta)
JYPPX_TRT11_LAYER_DOUBLE_GET(jyppx_trt11_fill_layer_get_beta, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getBeta, out_beta)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_fill_layer_set_to_type, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setToType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_fill_layer_get_to_type, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getToType)

JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_moe_layer_set_activation_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", setActivationType, nvinfer1::MoEActType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_moe_layer_get_activation_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", getActivationType)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_moe_layer_set_quantization_to_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", setQuantizationToType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_moe_layer_get_quantization_to_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", getQuantizationToType)
JYPPX_TRT11_LAYER_DIMS_SET(jyppx_trt11_moe_layer_set_quantization_block_shape, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", setQuantizationBlockShape, block_shape)
JYPPX_TRT11_LAYER_DIMS_GET(jyppx_trt11_moe_layer_get_quantization_block_shape, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", getQuantizationBlockShape, out_block_shape)
JYPPX_TRT11_LAYER_INT_SET(jyppx_trt11_moe_layer_set_dyn_q_output_scale_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", setDynQOutputScaleType, nvinfer1::DataType)
JYPPX_TRT11_LAYER_INT_GET(jyppx_trt11_moe_layer_get_dyn_q_output_scale_type, nvinfer1::IMoELayer, nvinfer1::LayerType::kMOE, "MoE", getDynQOutputScaleType)

JYPPX_StatusCode jyppx_trt11_moe_layer_set_gated_weights(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor* fc_gate_weights_tensor,
    JYPPX_TensorRtTensor* fc_up_weights_tensor,
    JYPPX_TensorRtTensor* fc_down_weights_tensor,
    int32_t activation_type)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IMoELayer* typed_layer = nullptr;
    nvinfer1::ITensor* fc_gate_weights = nullptr;
    nvinfer1::ITensor* fc_up_weights = nullptr;
    nvinfer1::ITensor* fc_down_weights = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE gated weights set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_gate_weights_tensor, &fc_gate_weights, "fc_gate_weights_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_up_weights_tensor, &fc_up_weights, "fc_up_weights_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_down_weights_tensor, &fc_down_weights, "fc_down_weights_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setGatedWeights(
        *fc_gate_weights,
        *fc_up_weights,
        *fc_down_weights,
        static_cast<nvinfer1::MoEActType>(activation_type));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)fc_gate_weights_tensor;
    (void)fc_up_weights_tensor;
    (void)fc_down_weights_tensor;
    (void)activation_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "MoE gated weights set");
#endif
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_gated_biases(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor* fc_gate_biases_tensor,
    JYPPX_TensorRtTensor* fc_up_biases_tensor,
    JYPPX_TensorRtTensor* fc_down_biases_tensor)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IMoELayer* typed_layer = nullptr;
    nvinfer1::ITensor* fc_gate_biases = nullptr;
    nvinfer1::ITensor* fc_up_biases = nullptr;
    nvinfer1::ITensor* fc_down_biases = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE gated biases set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_gate_biases_tensor, &fc_gate_biases, "fc_gate_biases_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_up_biases_tensor, &fc_up_biases, "fc_up_biases_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_down_biases_tensor, &fc_down_biases, "fc_down_biases_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setGatedBiases(*fc_gate_biases, *fc_up_biases, *fc_down_biases);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)fc_gate_biases_tensor;
    (void)fc_up_biases_tensor;
    (void)fc_down_biases_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "MoE gated biases set");
#endif
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_quantization_static(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor* fc_down_activation_scale_tensor,
    int32_t data_type)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IMoELayer* typed_layer = nullptr;
    nvinfer1::ITensor* fc_down_activation_scale = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE static quantization set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(fc_down_activation_scale_tensor, &fc_down_activation_scale, "fc_down_activation_scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setQuantizationStatic(*fc_down_activation_scale, static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)fc_down_activation_scale_tensor;
    (void)data_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "MoE static quantization set");
#endif
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_quantization_dynamic_dbl_q(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor* fc_down_activation_dbl_q_scale_tensor,
    int32_t data_type,
    const JYPPX_TensorRtDims* block_shape,
    int32_t dyn_q_output_scale_type)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IMoELayer* typed_layer = nullptr;
    nvinfer1::ITensor* fc_down_activation_dbl_q_scale = nullptr;
    nvinfer1::Dims native_block_shape{};
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE dynamic double quantization set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(
        fc_down_activation_dbl_q_scale_tensor,
        &fc_down_activation_dbl_q_scale,
        "fc_down_activation_dbl_q_scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = make_dims(block_shape, &native_block_shape, "block_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setQuantizationDynamicDblQ(
        *fc_down_activation_dbl_q_scale,
        static_cast<nvinfer1::DataType>(data_type),
        native_block_shape,
        static_cast<nvinfer1::DataType>(dyn_q_output_scale_type));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)fc_down_activation_dbl_q_scale_tensor;
    (void)data_type;
    (void)block_shape;
    (void)dyn_q_output_scale_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "MoE dynamic double quantization set");
#endif
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_swiglu_params(JYPPX_TensorRtLayer* layer, float limit, float alpha, float beta)
{
    nvinfer1::IMoELayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU params set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSwigluParams(limit, alpha, beta);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_swiglu_param_limit(JYPPX_TensorRtLayer* layer, float limit)
{
    nvinfer1::IMoELayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU limit set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSwigluParamLimit(limit);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_get_swiglu_param_limit(JYPPX_TensorRtLayer* layer, float* out_limit)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_limit, "out_limit");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_limit = 0.0F;
    nvinfer1::IMoELayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU limit query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_limit = typed_layer->getSwigluParamLimit();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_swiglu_param_alpha(JYPPX_TensorRtLayer* layer, float alpha)
{
    nvinfer1::IMoELayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU alpha set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSwigluParamAlpha(alpha);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_get_swiglu_param_alpha(JYPPX_TensorRtLayer* layer, float* out_alpha)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = 0.0F;
    nvinfer1::IMoELayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU alpha query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = typed_layer->getSwigluParamAlpha();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_set_swiglu_param_beta(JYPPX_TensorRtLayer* layer, float beta)
{
    nvinfer1::IMoELayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU beta set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSwigluParamBeta(beta);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_moe_layer_get_swiglu_param_beta(JYPPX_TensorRtLayer* layer, float* out_beta)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = 0.0F;
    nvinfer1::IMoELayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kMOE, "MoE", "MoE SwiGLU beta query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = typed_layer->getSwigluParamBeta();
    return JYPPX_STATUS_OK;
}

#undef JYPPX_TRT11_LAYER_BOOL_SET
#undef JYPPX_TRT11_LAYER_BOOL_GET
#undef JYPPX_TRT11_LAYER_INT64_SET
#undef JYPPX_TRT11_LAYER_INT64_GET
#undef JYPPX_TRT11_LAYER_DOUBLE_SET
#undef JYPPX_TRT11_LAYER_DOUBLE_GET
#undef JYPPX_TRT11_LAYER_DIMS_SET
#undef JYPPX_TRT11_LAYER_DIMS_GET
#undef JYPPX_TRT11_LAYER_UINT_SET
#undef JYPPX_TRT11_LAYER_UINT_GET
#undef JYPPX_TRT11_LAYER_INT_SET
#undef JYPPX_TRT11_LAYER_INT_GET
