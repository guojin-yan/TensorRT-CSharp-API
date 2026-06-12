#if JYPPX_HAS_TENSORRT

static JYPPX_StatusCode validate_dims64_dimension_index(const nvinfer1::Dims& dims, const int32_t dimension_index)
{
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (dimension_index >= dims.nbDims)
    {
        std::ostringstream builder;
        builder << "Tensor dimension index " << dimension_index << " is outside the tensor rank " << dims.nbDims << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode copy_dims64_dimension_extent(const nvinfer1::Dims& dims, const int32_t dimension_index, int64_t* out_extent)
{
    auto status = validate_dims64_dimension_index(dims, dimension_index);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = static_cast<int64_t>(dims.d[dimension_index]);
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_profile_payload_for_dims64_api(
    JYPPX_TensorRtOptimizationProfile* profile,
    nvinfer1::IOptimizationProfile** out_profile,
    const char* feature_name)
{
    *out_profile = nullptr;
    auto status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        std::ostringstream builder;
        builder << feature_name << " requires an optimization profile handle with a TensorRT payload.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_profile = profile_payload;
    return JYPPX_STATUS_OK;
}

#define JYPPX_TRT11_LAYER_DIMS_GET64(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims64* PARAM_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(PARAM_NAME, 0, sizeof(*PARAM_NAME)); \
        LAYER_TYPE* typed_layer = nullptr; \
        status = get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        copy_dims64(typed_layer->METHOD(), PARAM_NAME); \
        return JYPPX_STATUS_OK; \
    }

JYPPX_StatusCode jyppx_trt11_tensor_get_shape64(JYPPX_TensorRtTensor* tensor, JYPPX_TensorRtDims64* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_dims, 0, sizeof(*out_dims));
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(payload->getDimensions(), out_dims);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_tensor_get_dimension_extent64(JYPPX_TensorRtTensor* tensor, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor dimension extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(payload->getDimensions(), dimension_index, out_extent);
}

static JYPPX_StatusCode get_network_tensor_shape64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, bool output_slot, JYPPX_TensorRtDims64* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_shape, 0, sizeof(*out_shape));
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, output_slot ? "network output tensor shape64 query" : "network input tensor shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t count = output_slot ? network_payload->getNbOutputs() : network_payload->getNbInputs();
    status = validate_index(index, count, output_slot ? "output" : "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = output_slot ? network_payload->getOutput(index) : network_payload->getInput(index);
    if (tensor == nullptr) { return report_null_vendor_object(output_slot ? "INetworkDefinition::getOutput" : "INetworkDefinition::getInput"); }
    copy_dims64(tensor->getDimensions(), out_shape);
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_network_tensor_dimension_extent64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, int32_t dimension_index, bool output_slot, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, output_slot ? "network output tensor extent64 query" : "network input tensor extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t count = output_slot ? network_payload->getNbOutputs() : network_payload->getNbInputs();
    status = validate_index(index, count, output_slot ? "output" : "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor = output_slot ? network_payload->getOutput(index) : network_payload->getInput(index);
    if (tensor == nullptr) { return report_null_vendor_object(output_slot ? "INetworkDefinition::getOutput" : "INetworkDefinition::getInput"); }
    return copy_dims64_dimension_extent(tensor->getDimensions(), dimension_index, out_extent);
}

JYPPX_StatusCode jyppx_trt11_network_input_get_tensor_shape64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtDims64* out_shape)
{
    return get_network_tensor_shape64(network, index, false, out_shape);
}

JYPPX_StatusCode jyppx_trt11_network_output_get_tensor_shape64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtDims64* out_shape)
{
    return get_network_tensor_shape64(network, index, true, out_shape);
}

JYPPX_StatusCode jyppx_trt11_network_input_get_tensor_dimension_extent64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, int32_t dimension_index, int64_t* out_extent)
{
    return get_network_tensor_dimension_extent64(network, index, dimension_index, false, out_extent);
}

JYPPX_StatusCode jyppx_trt11_network_output_get_tensor_dimension_extent64(JYPPX_TensorRtNetworkDefinition* network, int32_t index, int32_t dimension_index, int64_t* out_extent)
{
    return get_network_tensor_dimension_extent64(network, index, dimension_index, true, out_extent);
}

#define JYPPX_TRT11_LAYER_SLOT_DIMS64(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtDims64* out_shape) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(out_shape, 0, sizeof(*out_shape)); \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        copy_dims64(tensor->getDimensions(), out_shape); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT64(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, int32_t dimension_index, int64_t* out_extent) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_extent = 0; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        return copy_dims64_dimension_extent(tensor->getDimensions(), dimension_index, out_extent); \
    }

JYPPX_TRT11_LAYER_SLOT_DIMS64(jyppx_trt11_layer_input_get_tensor_shape64, false)
JYPPX_TRT11_LAYER_SLOT_DIMS64(jyppx_trt11_layer_output_get_tensor_shape64, true)
JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT64(jyppx_trt11_layer_input_get_tensor_dimension_extent64, false)
JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT64(jyppx_trt11_layer_output_get_tensor_dimension_extent64, true)

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_shape64(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, JYPPX_TensorRtDims64* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_shape, 0, sizeof(*out_shape));
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(engine_payload->getTensorShape(tensor_name), out_shape);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_engine_get_tensor_dimension_extent64(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine tensor extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(engine_payload->getTensorShape(tensor_name), dimension_index, out_extent);
}

JYPPX_StatusCode jyppx_trt11_engine_get_profile_shape64(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, JYPPX_TensorRtDims64* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_shape, 0, sizeof(*out_shape));
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0 || selector < 0 || selector > 2) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(engine_payload->getProfileShape(tensor_name, profile_index, static_cast<nvinfer1::OptProfileSelector>(selector)), out_shape);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_engine_get_profile_shape_dimension_extent64(JYPPX_TensorRtCudaEngine* engine, const char* tensor_name, int32_t profile_index, int32_t selector, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0 || selector < 0 || selector > 2) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(engine_payload->getProfileShape(tensor_name, profile_index, static_cast<nvinfer1::OptProfileSelector>(selector)), dimension_index, out_extent);
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_shape64(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims64* out_shape)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_shape, 0, sizeof(*out_shape));
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(context_payload->getTensorShape(tensor_name), out_shape);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_shape_dimension_extent64(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(context_payload->getTensorShape(tensor_name), dimension_index, out_extent);
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_strides64(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_TensorRtDims64* out_strides)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_strides, "out_strides");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_strides, 0, sizeof(*out_strides));
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor strides64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(context_payload->getTensorStrides(tensor_name), out_strides);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_stride_dimension_extent64(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor stride extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(context_payload->getTensorStrides(tensor_name), dimension_index, out_extent);
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape64(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, JYPPX_TensorRtDims64* out_dims)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dims, "out_dims");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_dims, 0, sizeof(*out_dims));
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IOptimizationProfile* profile_payload = nullptr;
    status = get_profile_payload_for_dims64_api(profile, &profile_payload, "optimization profile shape64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_dims64(profile_payload->getDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector)), out_dims);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape_dimension_extent64(JYPPX_TensorRtOptimizationProfile* profile, const char* input_name, int32_t selector, int32_t dimension_index, int64_t* out_extent)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_extent = 0;
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IOptimizationProfile* profile_payload = nullptr;
    status = get_profile_payload_for_dims64_api(profile, &profile_payload, "optimization profile extent64 query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_dims64_dimension_extent(profile_payload->getDimensions(input_name, static_cast<nvinfer1::OptProfileSelector>(selector)), dimension_index, out_extent);
}

JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_shuffle_layer_get_reshape_dimensions64, nvinfer1::IShuffleLayer, nvinfer1::LayerType::kSHUFFLE, "shuffle", getReshapeDimensions, out_dims)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_slice_layer_get_start64, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getStart, out_start)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_slice_layer_get_size64, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getSize, out_size)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_slice_layer_get_stride64, nvinfer1::ISliceLayer, nvinfer1::LayerType::kSLICE, "slice", getStride, out_stride)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_dynamic_quantize_layer_get_block_shape64, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getBlockShape, out_block_shape)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_pooling_layer_get_window_size_nd64, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getWindowSizeNd, out_window_size)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_pooling_layer_get_stride_nd64, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getStrideNd, out_stride)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_pooling_layer_get_padding_nd64, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPaddingNd, out_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_pooling_layer_get_pre_padding64, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPrePadding, out_pre_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_pooling_layer_get_post_padding64, nvinfer1::IPoolingLayer, nvinfer1::LayerType::kPOOLING, "pooling", getPostPadding, out_post_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_resize_layer_get_output_dimensions64, nvinfer1::IResizeLayer, nvinfer1::LayerType::kRESIZE, "resize", getOutputDimensions, out_dimensions)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_fill_layer_get_dimensions64, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getDimensions, out_dimensions)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_convolution_layer_get_stride_nd64, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getStrideNd, out_stride)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_convolution_layer_get_pre_padding64, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getPrePadding, out_pre_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_convolution_layer_get_post_padding64, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getPostPadding, out_post_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_convolution_layer_get_dilation_nd64, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getDilationNd, out_dilation)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_deconvolution_layer_get_kernel_size_nd64, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getKernelSizeNd, out_kernel_size)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_deconvolution_layer_get_stride_nd64, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getStrideNd, out_stride)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_deconvolution_layer_get_pre_padding64, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getPrePadding, out_pre_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_deconvolution_layer_get_post_padding64, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getPostPadding, out_post_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_deconvolution_layer_get_dilation_nd64, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getDilationNd, out_dilation)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_padding_layer_get_pre_padding_nd64, nvinfer1::IPaddingLayer, nvinfer1::LayerType::kPADDING, "padding", getPrePaddingNd, out_pre_padding)
JYPPX_TRT11_LAYER_DIMS_GET64(jyppx_trt11_padding_layer_get_post_padding_nd64, nvinfer1::IPaddingLayer, nvinfer1::LayerType::kPADDING, "padding", getPostPaddingNd, out_post_padding)

#undef JYPPX_TRT11_LAYER_DIMS_GET64
#undef JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT64
#undef JYPPX_TRT11_LAYER_SLOT_DIMS64

#else

#define JYPPX_TRT11_STUB_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_TENSOR_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtTensor*, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_TENSOR_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtTensor*, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_NETWORK_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtNetworkDefinition*, int32_t, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_NETWORK_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtNetworkDefinition*, int32_t, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_LAYER_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_LAYER_SLOT_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_LAYER_SLOT_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_ENGINE_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_ENGINE_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_ENGINE_PROFILE_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, int32_t, int32_t, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_ENGINE_PROFILE_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtCudaEngine*, const char*, int32_t, int32_t, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_CONTEXT_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext*, const char*, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_CONTEXT_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtExecutionContext*, const char*, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_PROFILE_DIMS64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtOptimizationProfile*, const char*, int32_t, JYPPX_TensorRtDims64*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_STUB_PROFILE_EXTENT64(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtOptimizationProfile*, const char*, int32_t, int32_t, int64_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }

JYPPX_TRT11_STUB_TENSOR_DIMS64(jyppx_trt11_tensor_get_shape64)
JYPPX_TRT11_STUB_TENSOR_EXTENT64(jyppx_trt11_tensor_get_dimension_extent64)
JYPPX_TRT11_STUB_NETWORK_DIMS64(jyppx_trt11_network_input_get_tensor_shape64)
JYPPX_TRT11_STUB_NETWORK_DIMS64(jyppx_trt11_network_output_get_tensor_shape64)
JYPPX_TRT11_STUB_NETWORK_EXTENT64(jyppx_trt11_network_input_get_tensor_dimension_extent64)
JYPPX_TRT11_STUB_NETWORK_EXTENT64(jyppx_trt11_network_output_get_tensor_dimension_extent64)
JYPPX_TRT11_STUB_LAYER_SLOT_DIMS64(jyppx_trt11_layer_input_get_tensor_shape64)
JYPPX_TRT11_STUB_LAYER_SLOT_DIMS64(jyppx_trt11_layer_output_get_tensor_shape64)
JYPPX_TRT11_STUB_LAYER_SLOT_EXTENT64(jyppx_trt11_layer_input_get_tensor_dimension_extent64)
JYPPX_TRT11_STUB_LAYER_SLOT_EXTENT64(jyppx_trt11_layer_output_get_tensor_dimension_extent64)
JYPPX_TRT11_STUB_ENGINE_DIMS64(jyppx_trt11_engine_get_tensor_shape64)
JYPPX_TRT11_STUB_ENGINE_EXTENT64(jyppx_trt11_engine_get_tensor_dimension_extent64)
JYPPX_TRT11_STUB_ENGINE_PROFILE_DIMS64(jyppx_trt11_engine_get_profile_shape64)
JYPPX_TRT11_STUB_ENGINE_PROFILE_EXTENT64(jyppx_trt11_engine_get_profile_shape_dimension_extent64)
JYPPX_TRT11_STUB_CONTEXT_DIMS64(jyppx_trt11_execution_context_get_tensor_shape64)
JYPPX_TRT11_STUB_CONTEXT_EXTENT64(jyppx_trt11_execution_context_get_tensor_shape_dimension_extent64)
JYPPX_TRT11_STUB_CONTEXT_DIMS64(jyppx_trt11_execution_context_get_tensor_strides64)
JYPPX_TRT11_STUB_CONTEXT_EXTENT64(jyppx_trt11_execution_context_get_tensor_stride_dimension_extent64)
JYPPX_TRT11_STUB_PROFILE_DIMS64(jyppx_trt11_optimization_profile_get_shape64)
JYPPX_TRT11_STUB_PROFILE_EXTENT64(jyppx_trt11_optimization_profile_get_shape_dimension_extent64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_shuffle_layer_get_reshape_dimensions64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_slice_layer_get_start64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_slice_layer_get_size64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_slice_layer_get_stride64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_dynamic_quantize_layer_get_block_shape64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_pooling_layer_get_window_size_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_pooling_layer_get_stride_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_pooling_layer_get_padding_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_pooling_layer_get_pre_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_pooling_layer_get_post_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_resize_layer_get_output_dimensions64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_fill_layer_get_dimensions64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_convolution_layer_get_stride_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_convolution_layer_get_pre_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_convolution_layer_get_post_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_convolution_layer_get_dilation_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_deconvolution_layer_get_kernel_size_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_deconvolution_layer_get_stride_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_deconvolution_layer_get_pre_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_deconvolution_layer_get_post_padding64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_deconvolution_layer_get_dilation_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_padding_layer_get_pre_padding_nd64)
JYPPX_TRT11_STUB_LAYER_DIMS64(jyppx_trt11_padding_layer_get_post_padding_nd64)

#undef JYPPX_TRT11_STUB_PROFILE_EXTENT64
#undef JYPPX_TRT11_STUB_PROFILE_DIMS64
#undef JYPPX_TRT11_STUB_CONTEXT_EXTENT64
#undef JYPPX_TRT11_STUB_CONTEXT_DIMS64
#undef JYPPX_TRT11_STUB_ENGINE_PROFILE_EXTENT64
#undef JYPPX_TRT11_STUB_ENGINE_PROFILE_DIMS64
#undef JYPPX_TRT11_STUB_ENGINE_EXTENT64
#undef JYPPX_TRT11_STUB_ENGINE_DIMS64
#undef JYPPX_TRT11_STUB_LAYER_SLOT_EXTENT64
#undef JYPPX_TRT11_STUB_LAYER_SLOT_DIMS64
#undef JYPPX_TRT11_STUB_LAYER_DIMS64
#undef JYPPX_TRT11_STUB_NETWORK_EXTENT64
#undef JYPPX_TRT11_STUB_NETWORK_DIMS64
#undef JYPPX_TRT11_STUB_TENSOR_EXTENT64
#undef JYPPX_TRT11_STUB_TENSOR_DIMS64
#undef JYPPX_TRT11_STUB_DIMS64

#endif
