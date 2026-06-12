JYPPX_StatusCode jyppx_trt10_network_add_lrn(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t window_size, float alpha, float beta, float k, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

    if (window_size < 1 || window_size > 15 || (window_size % 2) == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "LRN window size must be an odd value in [1, 15].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ILRNLayer* layer = network_payload->addLRN(*tensor_payload, static_cast<int64_t>(window_size), alpha, beta, k);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)alpha;
    (void)beta;
    (void)k;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add LRN");
#endif
}

#define JYPPX_TRT10_LRN_FLOAT_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GETTER, SETTER, OUTPUT_NAME, INPUT_NAME, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, float* OUTPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUTPUT_NAME, #OUTPUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUTPUT_NAME = 0.0F; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* lrn_layer = get_lrn_layer_payload(layer); \
        if (lrn_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT LRN layer."); return JYPPX_STATUS_INVALID_STATE; } \
        *OUTPUT_NAME = lrn_layer->GETTER(); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, float INPUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* lrn_layer = get_lrn_layer_payload(layer); \
        if (lrn_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT LRN layer."); return JYPPX_STATUS_INVALID_STATE; } \
        lrn_layer->SETTER(INPUT_NAME); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode jyppx_trt10_lrn_layer_get_window_size(JYPPX_TensorRtLayer* layer, int32_t* out_window_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_window_size, "out_window_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_window_size = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* lrn_layer = get_lrn_layer_payload(layer);
    if (lrn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT LRN layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_window_size = static_cast<int32_t>(lrn_layer->getWindowSize());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_lrn_layer_set_window_size(JYPPX_TensorRtLayer* layer, int32_t window_size)
{
    if (window_size < 1 || window_size > 15 || (window_size % 2) == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "LRN window size must be an odd value in [1, 15].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* lrn_layer = get_lrn_layer_payload(layer);
    if (lrn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT LRN layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    lrn_layer->setWindowSize(static_cast<int64_t>(window_size));
    return JYPPX_STATUS_OK;
}

JYPPX_TRT10_LRN_FLOAT_ACCESSOR(jyppx_trt10_lrn_layer_get_alpha, jyppx_trt10_lrn_layer_set_alpha, getAlpha, setAlpha, out_alpha, alpha, "LRN alpha")
JYPPX_TRT10_LRN_FLOAT_ACCESSOR(jyppx_trt10_lrn_layer_get_beta, jyppx_trt10_lrn_layer_set_beta, getBeta, setBeta, out_beta, beta, "LRN beta")
JYPPX_TRT10_LRN_FLOAT_ACCESSOR(jyppx_trt10_lrn_layer_get_k, jyppx_trt10_lrn_layer_set_k, getK, setK, out_k, k, "LRN k")
#else
JYPPX_StatusCode jyppx_trt10_lrn_layer_get_window_size(JYPPX_TensorRtLayer*, int32_t* out_window_size) { auto status = jyppx::tensorrt::validate_output_pointer(out_window_size, "out_window_size"); if (status != JYPPX_STATUS_OK) { return status; } *out_window_size = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer window size query"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_set_window_size(JYPPX_TensorRtLayer*, int32_t) { return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer window size set"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_get_alpha(JYPPX_TensorRtLayer*, float* out_alpha) { auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha"); if (status != JYPPX_STATUS_OK) { return status; } *out_alpha = 0.0F; return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer alpha query"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_set_alpha(JYPPX_TensorRtLayer*, float) { return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer alpha set"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_get_beta(JYPPX_TensorRtLayer*, float* out_beta) { auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta"); if (status != JYPPX_STATUS_OK) { return status; } *out_beta = 0.0F; return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer beta query"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_set_beta(JYPPX_TensorRtLayer*, float) { return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer beta set"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_get_k(JYPPX_TensorRtLayer*, float* out_k) { auto status = jyppx::tensorrt::validate_output_pointer(out_k, "out_k"); if (status != JYPPX_STATUS_OK) { return status; } *out_k = 0.0F; return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer k query"); }
JYPPX_StatusCode jyppx_trt10_lrn_layer_set_k(JYPPX_TensorRtLayer*, float) { return jyppx::tensorrt::report_vendor_missing(kLine, "LRN layer k set"); }
#endif
#undef JYPPX_TRT10_LRN_FLOAT_ACCESSOR
