JYPPX_StatusCode jyppx_trt10_network_add_input(JYPPX_TensorRtNetworkDefinition* network, const char* name, int32_t data_type, const JYPPX_TensorRtDims* dims, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (data_type < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor data type must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_tensor = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network add input");
    }

    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::ITensor* tensor = network_payload->addInput(name, static_cast<nvinfer1::DataType>(data_type), native_dims);
    return create_tensor_reference_handle(tensor, out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add input");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_mark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network mark output");
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    network_payload->markOutput(*tensor_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network mark output");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_unmark_output(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network unmark output");
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    network_payload->unmarkOutput(*tensor_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unmark output");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_input_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = network_payload->getNbInputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network input count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_output_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = network_payload->getNbOutputs();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network output count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_layer_count(JYPPX_TensorRtNetworkDefinition* network, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = network_payload->getNbLayers();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network layer count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_layer(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, network_payload->getNbLayers(), "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return create_layer_reference_handle(network_payload->getLayer(index), out_layer);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network layer query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_name(JYPPX_TensorRtNetworkDefinition* network, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return copy_string_to_buffer(network_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network name query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_set_name(JYPPX_TensorRtNetworkDefinition* network, const char* name)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    network_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network name set");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_flags(JYPPX_TensorRtNetworkDefinition* network, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_flags = network_payload->getFlags();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network flags query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network flags query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_has_implicit_batch_dimension(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_has_implicit_batch_dimension)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_implicit_batch_dimension, "out_has_implicit_batch_dimension");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_implicit_batch_dimension = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const bool has_implicit_batch_dimension = network_payload->hasImplicitBatchDimension();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    *out_has_implicit_batch_dimension = has_implicit_batch_dimension ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network implicit batch dimension query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network implicit batch dimension query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_flag(JYPPX_TensorRtNetworkDefinition* network, int32_t flag, JYPPX_Boolean* out_enabled)
{
    if (flag < 0 || flag > 31)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network creation flag index must be in the range [0, 31].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = network_payload->getFlag(static_cast<nvinfer1::NetworkDefinitionCreationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "network flag query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network flag query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_input(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, network_payload->getNbInputs(), "input");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return create_tensor_reference_handle(network_payload->getInput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network input query");
#endif
}

JYPPX_StatusCode jyppx_trt10_network_get_output(JYPPX_TensorRtNetworkDefinition* network, int32_t index, JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor = nullptr;

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = validate_index(index, network_payload->getNbOutputs(), "output");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return create_tensor_reference_handle(network_payload->getOutput(index), out_tensor);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network output query");
#endif
}
