JYPPX_StatusCode jyppx_trt8_network_add_deconvolution_nd(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t output_maps, const JYPPX_TensorRtDims* kernel_size, int32_t data_type, const void* kernel_values, size_t kernel_value_count, const void* bias_values, size_t bias_value_count, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (output_maps <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Deconvolution output maps must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    nvinfer1::Dims native_kernel{};
    status = make_dims(kernel_size, &native_kernel, "kernel_size");
    if (status != JYPPX_STATUS_OK) { return status; }

    std::vector<uint8_t> kernel_blob;
    status = copy_optional_weights(data_type, kernel_values, kernel_value_count, "Deconvolution kernel", true, &kernel_blob);
    if (status != JYPPX_STATUS_OK) { return status; }

    std::vector<uint8_t> bias_blob;
    status = copy_optional_weights(data_type, bias_values, bias_value_count, "Deconvolution bias", false, &bias_blob);
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::Weights kernel_weights{static_cast<nvinfer1::DataType>(data_type), kernel_blob.data(), static_cast<int64_t>(kernel_value_count)};
    nvinfer1::Weights bias_weights{static_cast<nvinfer1::DataType>(data_type), bias_blob.empty() ? nullptr : bias_blob.data(), static_cast<int64_t>(bias_value_count)};
    nvinfer1::IDeconvolutionLayer* layer = network_payload->addDeconvolutionNd(*tensor_payload, output_maps, native_kernel, kernel_weights, bias_weights);

    std::vector<std::vector<uint8_t>> owned_blobs;
    owned_blobs.push_back(std::move(kernel_blob));
    if (!bias_blob.empty()) { owned_blobs.push_back(std::move(bias_blob)); }
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, 0, -1, -1, 0, 0, 0, std::move(owned_blobs));
#else
    (void)network;
    (void)input_tensor;
    (void)kernel_size;
    (void)data_type;
    (void)kernel_values;
    (void)kernel_value_count;
    (void)bias_values;
    (void)bias_value_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add deconvolution");
#endif
}

#define JYPPX_TRT8_DECONV_INT_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, INPUT_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, int32_t* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = 0; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* deconvolution_layer = get_deconvolution_layer_payload(layer); \
        if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; } \
        *OUTPUT_NAME = deconvolution_layer->GETTER(); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, int32_t INPUT_NAME) \
    { \
        if (INPUT_NAME <= 0) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, FEATURE " must be greater than zero."); return JYPPX_STATUS_INVALID_ARGUMENT; } \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* deconvolution_layer = get_deconvolution_layer_payload(layer); \
        if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; } \
        deconvolution_layer->SETTER(INPUT_NAME); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT8_DECONV_INT_ACCESSOR(jyppx_trt8_deconvolution_layer_get_nb_output_maps, jyppx_trt8_deconvolution_layer_set_nb_output_maps, getNbOutputMaps, setNbOutputMaps, out_output_maps, output_maps, "Deconvolution output maps")
JYPPX_TRT8_DECONV_INT_ACCESSOR(jyppx_trt8_deconvolution_layer_get_nb_groups, jyppx_trt8_deconvolution_layer_set_nb_groups, getNbGroups, setNbGroups, out_groups, groups, "Deconvolution groups")
#else
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_nb_output_maps(JYPPX_TensorRtLayer*, int32_t* out_output_maps) { auto status = jyppx::tensorrt::validate_output_pointer(out_output_maps, "out_output_maps"); if (status != JYPPX_STATUS_OK) { return status; } *out_output_maps = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer output map query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_nb_output_maps(JYPPX_TensorRtLayer*, int32_t) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer output map set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_nb_groups(JYPPX_TensorRtLayer*, int32_t* out_groups) { auto status = jyppx::tensorrt::validate_output_pointer(out_groups, "out_groups"); if (status != JYPPX_STATUS_OK) { return status; } *out_groups = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer group query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_nb_groups(JYPPX_TensorRtLayer*, int32_t) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer group set"); }
#endif
#undef JYPPX_TRT8_DECONV_INT_ACCESSOR

#define JYPPX_TRT8_DECONV_DIMS_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, INPUT_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(OUTPUT_NAME, 0, sizeof(*OUTPUT_NAME)); \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* deconvolution_layer = get_deconvolution_layer_payload(layer); \
        if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; } \
        copy_dims(deconvolution_layer->GETTER(), OUTPUT_NAME); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* INPUT_NAME) \
    { \
        nvinfer1::Dims native_dims{}; \
        auto status = make_dims(INPUT_NAME, &native_dims, #INPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* deconvolution_layer = get_deconvolution_layer_payload(layer); \
        if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; } \
        deconvolution_layer->SETTER(native_dims); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_kernel_size_nd, jyppx_trt8_deconvolution_layer_set_kernel_size_nd, getKernelSizeNd, setKernelSizeNd, out_kernel_size, kernel_size, "deconvolution kernel size")
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_stride_nd, jyppx_trt8_deconvolution_layer_set_stride_nd, getStrideNd, setStrideNd, out_stride, stride, "deconvolution stride")
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_padding_nd, jyppx_trt8_deconvolution_layer_set_padding_nd, getPaddingNd, setPaddingNd, out_padding, padding, "deconvolution padding")
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_pre_padding, jyppx_trt8_deconvolution_layer_set_pre_padding, getPrePadding, setPrePadding, out_pre_padding, pre_padding, "deconvolution pre-padding")
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_post_padding, jyppx_trt8_deconvolution_layer_set_post_padding, getPostPadding, setPostPadding, out_post_padding, post_padding, "deconvolution post-padding")
JYPPX_TRT8_DECONV_DIMS_ACCESSOR(jyppx_trt8_deconvolution_layer_get_dilation_nd, jyppx_trt8_deconvolution_layer_set_dilation_nd, getDilationNd, setDilationNd, out_dilation, dilation, "deconvolution dilation")
#else
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_kernel_size_nd(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_kernel_size) { auto status = jyppx::tensorrt::validate_output_pointer(out_kernel_size, "out_kernel_size"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_kernel_size, 0, sizeof(*out_kernel_size)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer kernel size query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_kernel_size_nd(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer kernel size set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_stride_nd(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_stride) { auto status = jyppx::tensorrt::validate_output_pointer(out_stride, "out_stride"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_stride, 0, sizeof(*out_stride)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer stride query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_stride_nd(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer stride set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_padding_nd(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_padding) { auto status = jyppx::tensorrt::validate_output_pointer(out_padding, "out_padding"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_padding, 0, sizeof(*out_padding)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer padding query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_padding_nd(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer padding set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_pre_padding(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_pre_padding) { auto status = jyppx::tensorrt::validate_output_pointer(out_pre_padding, "out_pre_padding"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_pre_padding, 0, sizeof(*out_pre_padding)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer pre-padding query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_pre_padding(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer pre-padding set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_post_padding(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_post_padding) { auto status = jyppx::tensorrt::validate_output_pointer(out_post_padding, "out_post_padding"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_post_padding, 0, sizeof(*out_post_padding)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer post-padding query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_post_padding(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer post-padding set"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_dilation_nd(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* out_dilation) { auto status = jyppx::tensorrt::validate_output_pointer(out_dilation, "out_dilation"); if (status != JYPPX_STATUS_OK) { return status; } std::memset(out_dilation, 0, sizeof(*out_dilation)); return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer dilation query"); }
JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_dilation_nd(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer dilation set"); }
#endif
#undef JYPPX_TRT8_DECONV_DIMS_ACCESSOR

JYPPX_StatusCode jyppx_trt8_deconvolution_layer_get_padding_mode(JYPPX_TensorRtLayer* layer, int32_t* out_padding_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_padding_mode, "out_padding_mode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_padding_mode = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* deconvolution_layer = get_deconvolution_layer_payload(layer);
    if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; }
    *out_padding_mode = static_cast<int32_t>(deconvolution_layer->getPaddingMode());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer padding mode query");
#endif
}

JYPPX_StatusCode jyppx_trt8_deconvolution_layer_set_padding_mode(JYPPX_TensorRtLayer* layer, int32_t padding_mode)
{
    if (padding_mode < 0 || padding_mode > 3)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Padding mode must be in the TensorRT enum range [0, 3].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* deconvolution_layer = get_deconvolution_layer_payload(layer);
    if (deconvolution_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT deconvolution layer."); return JYPPX_STATUS_INVALID_STATE; }
    deconvolution_layer->setPaddingMode(static_cast<nvinfer1::PaddingMode>(padding_mode));
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "deconvolution layer padding mode set");
#endif
}
