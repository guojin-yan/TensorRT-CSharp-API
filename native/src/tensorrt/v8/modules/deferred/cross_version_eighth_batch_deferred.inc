#define JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(function_name, interface_name, reason) \
    JYPPX_StatusCode function_name(void) \
    { \
        return jyppx::tensorrt::report_not_implemented(kLine, interface_name " deferred: " reason); \
    }

JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_destroy_deferred, "IOnnxConfig::destroy", "legacy ONNX config handles are not modeled; vendor destroy is unsafe without bridge-owned lifetime")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_full_text_file_name_deferred, "IOnnxConfig::getFullTextFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_model_dtype_deferred, "IOnnxConfig::getModelDtype", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_model_file_name_deferred, "IOnnxConfig::getModelFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_print_layer_info_deferred, "IOnnxConfig::getPrintLayerInfo", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_text_file_name_deferred, "IOnnxConfig::getTextFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_get_verbosity_level_deferred, "IOnnxConfig::getVerbosityLevel", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_reduce_verbosity_deferred, "IOnnxConfig::reduceVerbosity", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_full_text_file_name_deferred, "IOnnxConfig::setFullTextFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_model_dtype_deferred, "IOnnxConfig::setModelDtype", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_model_file_name_deferred, "IOnnxConfig::setModelFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_print_layer_info_deferred, "IOnnxConfig::setPrintLayerInfo", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_text_file_name_deferred, "IOnnxConfig::setTextFileName", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_onnx_config_set_verbosity_level_deferred, "IOnnxConfig::setVerbosityLevel", "legacy ONNX config handles are not modeled by the managed bridge")
JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB(jyppx_trt8_parser_parse_with_weight_descriptors_deferred, "IParser::parseWithWeightDescriptors", "external weight descriptor ownership and raw memory lifetime require a dedicated safe model")

#undef JYPPX_TRT8_EIGHTH_BATCH_DEFERRED_STUB
