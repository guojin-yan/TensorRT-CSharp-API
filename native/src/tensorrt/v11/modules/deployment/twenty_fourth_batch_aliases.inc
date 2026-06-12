JYPPX_StatusCode jyppx_trt11_builder_config_get_nb_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)
{
    return jyppx_trt11_builder_config_get_plugin_to_serialize_count(config, out_count);
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_plugin_to_serialize_v2(JYPPX_TensorRtBuilderConfig* config, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_builder_config_get_plugin_to_serialize(config, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_builder_config_clear_plugins_to_serialize_v2(JYPPX_TensorRtBuilderConfig* config)
{
    return jyppx_trt11_builder_config_clear_plugins_to_serialize(config);
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_plugins_to_serialize_v2(JYPPX_TensorRtBuilderConfig* config, const char** paths, int32_t path_count, JYPPX_Boolean* out_set)
{
    return jyppx_trt11_builder_config_set_plugins_to_serialize(config, paths, path_count, out_set);
}

JYPPX_StatusCode jyppx_trt11_builder_config_can_run_on_dla_v2(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_can_run)
{
    return jyppx_trt11_builder_config_can_run_on_dla(config, layer, out_can_run);
}

JYPPX_StatusCode jyppx_trt11_parser_get_nb_errors(JYPPX_TensorRtOnnxParser* parser, int32_t* out_count)
{
    return jyppx_trt11_onnx_parser_get_error_count(parser, out_count);
}

JYPPX_StatusCode jyppx_trt11_parser_clear_errors(JYPPX_TensorRtOnnxParser* parser)
{
    return jyppx_trt11_onnx_parser_clear_errors(parser);
}

JYPPX_StatusCode jyppx_trt11_parser_get_error(JYPPX_TensorRtOnnxParser* parser, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)
{
    return jyppx_trt11_onnx_parser_get_error(parser, index, out_error);
}

JYPPX_StatusCode jyppx_trt11_parser_get_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t* out_flags)
{
    return jyppx_trt11_onnx_parser_get_flags(parser, out_flags);
}

JYPPX_StatusCode jyppx_trt11_parser_set_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t flags)
{
    return jyppx_trt11_onnx_parser_set_flags(parser, flags);
}

JYPPX_StatusCode jyppx_trt11_parser_get_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag, JYPPX_Boolean* out_enabled)
{
    return jyppx_trt11_onnx_parser_get_flag(parser, flag, out_enabled);
}

JYPPX_StatusCode jyppx_trt11_parser_set_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    return jyppx_trt11_onnx_parser_set_flag(parser, flag);
}

JYPPX_StatusCode jyppx_trt11_parser_clear_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    return jyppx_trt11_onnx_parser_clear_flag(parser, flag);
}

JYPPX_StatusCode jyppx_trt11_parser_supports_operator(JYPPX_TensorRtOnnxParser* parser, const char* operator_name, JYPPX_Boolean* out_supported)
{
    return jyppx_trt11_onnx_parser_supports_operator(parser, operator_name, out_supported);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_code(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_code)
{
    return jyppx_trt11_onnx_parser_error_get_code(parser, index, out_code);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_line(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_line)
{
    return jyppx_trt11_onnx_parser_error_get_line(parser, index, out_line);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_node(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_node)
{
    return jyppx_trt11_onnx_parser_error_get_node(parser, index, out_node);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_description(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_onnx_parser_error_get_description(parser, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_file(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_onnx_parser_error_get_file(parser, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_function(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_onnx_parser_error_get_function(parser, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_node_name(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_onnx_parser_error_get_node_name(parser, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_parser_error_get_node_operator(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return jyppx_trt11_onnx_parser_error_get_node_operator(parser, index, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_missing_weights_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    return jyppx_trt11_refitter_get_missing_count(refitter, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_all_weights_count(JYPPX_TensorRtRefitter* refitter, int32_t* out_count)
{
    return jyppx_trt11_refitter_get_all_count(refitter, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_missing_weights_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    return jyppx_trt11_refitter_get_missing_entries(refitter, output_entries, output_count, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_all_weights_entries(JYPPX_TensorRtRefitter* refitter, JYPPX_TensorRtRefitEntryInfo* output_entries, int32_t output_count, int32_t* out_count)
{
    return jyppx_trt11_refitter_get_all_entries(refitter, output_entries, output_count, out_count);
}

JYPPX_StatusCode jyppx_trt11_refitter_refit(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_refitted)
{
    return jyppx_trt11_refitter_refit_cuda_engine(refitter, out_refitted);
}

JYPPX_StatusCode jyppx_trt11_refitter_refit_async(JYPPX_TensorRtRefitter* refitter, JYPPX_CudaStream* stream, JYPPX_Boolean* out_refitted)
{
    return jyppx_trt11_refitter_refit_cuda_engine_async(refitter, stream, out_refitted);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_weights_validation_v2(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_enabled)
{
    return jyppx_trt11_refitter_get_weights_validation(refitter, out_enabled);
}

JYPPX_StatusCode jyppx_trt11_refitter_set_weights_validation_v2(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean enabled)
{
    return jyppx_trt11_refitter_set_weights_validation(refitter, enabled);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_named_weights_location(JYPPX_TensorRtRefitter* refitter, const char* weights_name, int32_t* out_location)
{
    return jyppx_trt11_refitter_get_weights_location(refitter, weights_name, out_location);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_named_weights_info_v2(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info)
{
    return jyppx_trt11_refitter_get_named_weights_info(refitter, weights_name, out_info);
}

JYPPX_StatusCode jyppx_trt11_refitter_get_weights_prototype_info_v2(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info)
{
    return jyppx_trt11_refitter_get_weights_prototype_info(refitter, weights_name, out_info);
}
