#if JYPPX_HAS_TENSORRT

static JYPPX_StatusCode get_layer_slot_tensor_for_api(
    JYPPX_TensorRtLayer* layer,
    const int32_t index,
    const bool output_slot,
    const bool require_tensor,
    nvinfer1::ITensor** out_tensor,
    const char* feature_name)
{
    *out_tensor = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    auto status = get_layer_payload_for_api(layer, &layer_payload, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    const int32_t count = output_slot ? layer_payload->getNbOutputs() : layer_payload->getNbInputs();
    status = validate_index(index, count, output_slot ? "output" : "input");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::ITensor* tensor = output_slot ? layer_payload->getOutput(index) : layer_payload->getInput(index);
    if (tensor == nullptr && require_tensor)
    {
        std::ostringstream builder;
        builder << "Layer " << (output_slot ? "output" : "input") << " slot " << index << " does not contain a TensorRT tensor.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_tensor = tensor;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode validate_layer_tensor_dimension_index(
    const nvinfer1::ITensor* tensor,
    const int32_t dimension_index)
{
    if (dimension_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor dimension index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const auto dims = tensor->getDimensions();
    if (dimension_index >= dims.nbDims)
    {
        std::ostringstream builder;
        builder << "Tensor dimension index " << dimension_index << " is outside the tensor rank " << dims.nbDims << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

static bool tensor_has_dynamic_dimension(const nvinfer1::ITensor* tensor)
{
    const auto dims = tensor->getDimensions();
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (dims.d[index] < 0)
        {
            return true;
        }
    }

    return false;
}

static JYPPX_StatusCode copy_layer_tensor_dimension_extent(
    const nvinfer1::ITensor* tensor,
    const int32_t dimension_index,
    int32_t* out_extent)
{
    const auto extent = tensor->getDimensions().d[dimension_index];
    if (extent < std::numeric_limits<int32_t>::min() || extent > std::numeric_limits<int32_t>::max())
    {
        std::ostringstream builder;
        builder << "Tensor dimension extent " << extent << " is outside the int32 ABI range.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_extent = static_cast<int32_t>(extent);
    return JYPPX_STATUS_OK;
}

static void append_tensor_dims(std::ostringstream& builder, const nvinfer1::Dims& dims)
{
    builder << "[";
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (index > 0)
        {
            builder << ",";
        }

        builder << dims.d[index];
    }

    builder << "]";
}

static JYPPX_StatusCode copy_layer_slot_tensor_summary(
    JYPPX_TensorRtLayer* layer,
    const int32_t index,
    const bool output_slot,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::ITensor* tensor = nullptr;
    status = get_layer_slot_tensor_for_api(layer, index, output_slot, false, &tensor, output_slot ? "layer output tensor summary query" : "layer input tensor summary query");
    if (status != JYPPX_STATUS_OK) { return status; }

    if (tensor == nullptr)
    {
        return copy_string_to_buffer(output_slot ? "output=null" : "input=null", output_buffer, output_buffer_size, out_required_size);
    }

    const auto dims = tensor->getDimensions();
    std::ostringstream builder;
    builder
        << (output_slot ? "output" : "input")
        << "[" << index << "]"
        << " name=" << (tensor->getName() == nullptr ? "" : tensor->getName())
        << " type=" << static_cast<int32_t>(tensor->getType())
        << " dims=";
    append_tensor_dims(builder, dims);
    builder
        << " rank=" << dims.nbDims
        << " location=" << static_cast<int32_t>(tensor->getLocation())
        << " formats=" << static_cast<uint32_t>(tensor->getAllowedFormats())
        << " shape=" << (tensor->isShapeTensor() ? 1 : 0)
        << " execution=" << (tensor->isExecutionTensor() ? 1 : 0)
        << " networkInput=" << (tensor->isNetworkInput() ? 1 : 0)
        << " networkOutput=" << (tensor->isNetworkOutput() ? 1 : 0)
        << " dynamic=" << (tensor_has_dynamic_dimension(tensor) ? 1 : 0);

    return copy_string_to_buffer(builder.str().c_str(), output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_tensor_is_network_input(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_input)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_network_input, "out_is_network_input");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_network_input = JYPPX_FALSE;
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor network-input query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_network_input = payload->isNetworkInput() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_tensor_is_network_output(JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_network_output)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_network_output, "out_is_network_output");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_network_output = JYPPX_FALSE;
    nvinfer1::ITensor* payload = nullptr;
    status = get_tensor_payload_for_api(tensor, &payload, "tensor network-output query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_network_output = payload->isNetworkOutput() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

#define JYPPX_TRT11_LAYER_SLOT_HAS(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_Boolean* out_has_tensor) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_has_tensor, "out_has_tensor"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_has_tensor = JYPPX_FALSE; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, false, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_has_tensor = tensor == nullptr ? JYPPX_FALSE : JYPPX_TRUE; \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_STRING(FUNCTION_NAME, OUTPUT_SLOT, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        const char* value = (EXPR); \
        return copy_string_to_buffer(value == nullptr ? "" : value, output_buffer, output_buffer_size, out_required_size); \
    }

#define JYPPX_TRT11_LAYER_SLOT_INT(FUNCTION_NAME, OUTPUT_SLOT, OUT_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, int32_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<int32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_UINT(FUNCTION_NAME, OUTPUT_SLOT, OUT_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, uint32_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<uint32_t>(EXPR); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_BOOL(FUNCTION_NAME, OUTPUT_SLOT, OUT_NAME, EXPR) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_Boolean* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = JYPPX_FALSE; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = (EXPR) ? JYPPX_TRUE : JYPPX_FALSE; \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_DIMS(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtDims* out_shape) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_shape, "out_shape"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(out_shape, 0, sizeof(*out_shape)); \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        copy_dims(tensor->getDimensions(), out_shape); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, int32_t dimension_index, int32_t* out_extent) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_extent, "out_extent"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_extent = 0; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        status = validate_layer_tensor_dimension_index(tensor, dimension_index); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        return copy_layer_tensor_dimension_extent(tensor, dimension_index, out_extent); \
    }

#define JYPPX_TRT11_LAYER_SLOT_DIM_HAS_NAME(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, int32_t dimension_index, JYPPX_Boolean* out_has_name) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_has_name, "out_has_name"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_has_name = JYPPX_FALSE; \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        status = validate_layer_tensor_dimension_index(tensor, dimension_index); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        const char* name = tensor->getDimensionName(dimension_index); \
        *out_has_name = (name != nullptr && name[0] != '\0') ? JYPPX_TRUE : JYPPX_FALSE; \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT11_LAYER_SLOT_DIM_NAME(FUNCTION_NAME, OUTPUT_SLOT) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t index, int32_t dimension_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        nvinfer1::ITensor* tensor = nullptr; \
        status = get_layer_slot_tensor_for_api(layer, index, OUTPUT_SLOT, true, &tensor, #FUNCTION_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        status = validate_layer_tensor_dimension_index(tensor, dimension_index); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        const char* name = tensor->getDimensionName(dimension_index); \
        return copy_string_to_buffer(name == nullptr ? "" : name, output_buffer, output_buffer_size, out_required_size); \
    }

JYPPX_TRT11_LAYER_SLOT_HAS(jyppx_trt11_layer_input_has_tensor, false)
JYPPX_TRT11_LAYER_SLOT_HAS(jyppx_trt11_layer_output_has_tensor, true)
JYPPX_TRT11_LAYER_SLOT_STRING(jyppx_trt11_layer_input_get_tensor_name, false, tensor->getName())
JYPPX_TRT11_LAYER_SLOT_STRING(jyppx_trt11_layer_output_get_tensor_name, true, tensor->getName())
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_input_get_tensor_data_type, false, out_data_type, tensor->getType())
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_output_get_tensor_data_type, true, out_data_type, tensor->getType())
JYPPX_TRT11_LAYER_SLOT_DIMS(jyppx_trt11_layer_input_get_tensor_shape, false)
JYPPX_TRT11_LAYER_SLOT_DIMS(jyppx_trt11_layer_output_get_tensor_shape, true)
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_input_get_tensor_dimension_count, false, out_count, tensor->getDimensions().nbDims)
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_output_get_tensor_dimension_count, true, out_count, tensor->getDimensions().nbDims)
JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT(jyppx_trt11_layer_input_get_tensor_dimension_extent, false)
JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT(jyppx_trt11_layer_output_get_tensor_dimension_extent, true)
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_input_tensor_has_dynamic_dimension, false, out_has_dynamic_dimension, tensor_has_dynamic_dimension(tensor))
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_output_tensor_has_dynamic_dimension, true, out_has_dynamic_dimension, tensor_has_dynamic_dimension(tensor))
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_input_get_tensor_location, false, out_location, tensor->getLocation())
JYPPX_TRT11_LAYER_SLOT_INT(jyppx_trt11_layer_output_get_tensor_location, true, out_location, tensor->getLocation())
JYPPX_TRT11_LAYER_SLOT_UINT(jyppx_trt11_layer_input_get_tensor_allowed_formats, false, out_formats, tensor->getAllowedFormats())
JYPPX_TRT11_LAYER_SLOT_UINT(jyppx_trt11_layer_output_get_tensor_allowed_formats, true, out_formats, tensor->getAllowedFormats())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_input_tensor_is_shape_tensor, false, out_is_shape_tensor, tensor->isShapeTensor())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_output_tensor_is_shape_tensor, true, out_is_shape_tensor, tensor->isShapeTensor())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_input_tensor_is_execution_tensor, false, out_is_execution_tensor, tensor->isExecutionTensor())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_output_tensor_is_execution_tensor, true, out_is_execution_tensor, tensor->isExecutionTensor())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_input_tensor_is_network_input, false, out_is_network_input, tensor->isNetworkInput())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_output_tensor_is_network_input, true, out_is_network_input, tensor->isNetworkInput())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_input_tensor_is_network_output, false, out_is_network_output, tensor->isNetworkOutput())
JYPPX_TRT11_LAYER_SLOT_BOOL(jyppx_trt11_layer_output_tensor_is_network_output, true, out_is_network_output, tensor->isNetworkOutput())
JYPPX_TRT11_LAYER_SLOT_DIM_HAS_NAME(jyppx_trt11_layer_input_tensor_dimension_has_name, false)
JYPPX_TRT11_LAYER_SLOT_DIM_HAS_NAME(jyppx_trt11_layer_output_tensor_dimension_has_name, true)
JYPPX_TRT11_LAYER_SLOT_DIM_NAME(jyppx_trt11_layer_input_get_tensor_dimension_name, false)
JYPPX_TRT11_LAYER_SLOT_DIM_NAME(jyppx_trt11_layer_output_get_tensor_dimension_name, true)

JYPPX_StatusCode jyppx_trt11_layer_input_get_tensor_summary(
    JYPPX_TensorRtLayer* layer,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return copy_layer_slot_tensor_summary(layer, index, false, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_layer_output_get_tensor_summary(
    JYPPX_TensorRtLayer* layer,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return copy_layer_slot_tensor_summary(layer, index, true, output_buffer, output_buffer_size, out_required_size);
}

#undef JYPPX_TRT11_LAYER_SLOT_DIM_NAME
#undef JYPPX_TRT11_LAYER_SLOT_DIM_HAS_NAME
#undef JYPPX_TRT11_LAYER_SLOT_DIM_EXTENT
#undef JYPPX_TRT11_LAYER_SLOT_DIMS
#undef JYPPX_TRT11_LAYER_SLOT_BOOL
#undef JYPPX_TRT11_LAYER_SLOT_UINT
#undef JYPPX_TRT11_LAYER_SLOT_INT
#undef JYPPX_TRT11_LAYER_SLOT_STRING
#undef JYPPX_TRT11_LAYER_SLOT_HAS

#else

JYPPX_StatusCode jyppx_trt11_tensor_is_network_input(JYPPX_TensorRtTensor*, JYPPX_Boolean*) { return jyppx::tensorrt::report_vendor_missing(kLine, "tensor network-input query"); }
JYPPX_StatusCode jyppx_trt11_tensor_is_network_output(JYPPX_TensorRtTensor*, JYPPX_Boolean*) { return jyppx::tensorrt::report_vendor_missing(kLine, "tensor network-output query"); }

#define JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, JYPPX_Boolean*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_INT(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, int32_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_UINT(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, uint32_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_DIMS(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, JYPPX_TensorRtDims*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_STRING(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, char*, size_t, size_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_DIM_INT(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, int32_t, int32_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_DIM_BOOL(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, int32_t, JYPPX_Boolean*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }
#define JYPPX_TRT11_LAYER_SLOT_STUB_DIM_STRING(FUNCTION_NAME) JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t, int32_t, char*, size_t, size_t*) { return jyppx::tensorrt::report_vendor_missing(kLine, #FUNCTION_NAME); }

JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_has_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_has_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_STRING(jyppx_trt11_layer_input_get_tensor_name)
JYPPX_TRT11_LAYER_SLOT_STUB_STRING(jyppx_trt11_layer_output_get_tensor_name)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_input_get_tensor_data_type)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_output_get_tensor_data_type)
JYPPX_TRT11_LAYER_SLOT_STUB_DIMS(jyppx_trt11_layer_input_get_tensor_shape)
JYPPX_TRT11_LAYER_SLOT_STUB_DIMS(jyppx_trt11_layer_output_get_tensor_shape)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_input_get_tensor_dimension_count)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_output_get_tensor_dimension_count)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_INT(jyppx_trt11_layer_input_get_tensor_dimension_extent)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_INT(jyppx_trt11_layer_output_get_tensor_dimension_extent)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_tensor_has_dynamic_dimension)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_tensor_has_dynamic_dimension)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_input_get_tensor_location)
JYPPX_TRT11_LAYER_SLOT_STUB_INT(jyppx_trt11_layer_output_get_tensor_location)
JYPPX_TRT11_LAYER_SLOT_STUB_UINT(jyppx_trt11_layer_input_get_tensor_allowed_formats)
JYPPX_TRT11_LAYER_SLOT_STUB_UINT(jyppx_trt11_layer_output_get_tensor_allowed_formats)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_tensor_is_shape_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_tensor_is_shape_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_tensor_is_execution_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_tensor_is_execution_tensor)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_tensor_is_network_input)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_tensor_is_network_input)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_input_tensor_is_network_output)
JYPPX_TRT11_LAYER_SLOT_STUB_BOOL(jyppx_trt11_layer_output_tensor_is_network_output)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_BOOL(jyppx_trt11_layer_input_tensor_dimension_has_name)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_BOOL(jyppx_trt11_layer_output_tensor_dimension_has_name)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_STRING(jyppx_trt11_layer_input_get_tensor_dimension_name)
JYPPX_TRT11_LAYER_SLOT_STUB_DIM_STRING(jyppx_trt11_layer_output_get_tensor_dimension_name)
JYPPX_TRT11_LAYER_SLOT_STUB_STRING(jyppx_trt11_layer_input_get_tensor_summary)
JYPPX_TRT11_LAYER_SLOT_STUB_STRING(jyppx_trt11_layer_output_get_tensor_summary)

#undef JYPPX_TRT11_LAYER_SLOT_STUB_DIM_STRING
#undef JYPPX_TRT11_LAYER_SLOT_STUB_DIM_BOOL
#undef JYPPX_TRT11_LAYER_SLOT_STUB_DIM_INT
#undef JYPPX_TRT11_LAYER_SLOT_STUB_STRING
#undef JYPPX_TRT11_LAYER_SLOT_STUB_DIMS
#undef JYPPX_TRT11_LAYER_SLOT_STUB_UINT
#undef JYPPX_TRT11_LAYER_SLOT_STUB_INT
#undef JYPPX_TRT11_LAYER_SLOT_STUB_BOOL

#endif
