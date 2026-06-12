#define JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(function_name, interface_name, reason) \
    JYPPX_StatusCode function_name(void) \
    { \
        return jyppx::tensorrt::report_not_implemented(kLine, interface_name " deferred: " reason); \
    }

JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_dec_ref_count_deferred, "IErrorRecorder::decRefCount", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_enum_max_deferred, "IErrorRecorder::EnumMax", "error-recorder enum boundary is covered as deferred metadata until recorder ownership is modeled")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_get_error_code_deferred, "IErrorRecorder::getErrorCode", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_get_error_desc_deferred, "IErrorRecorder::getErrorDesc", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_get_nb_errors_deferred, "IErrorRecorder::getNbErrors", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_has_overflowed_deferred, "IErrorRecorder::hasOverflowed", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_error_recorder_inc_ref_count_deferred, "IErrorRecorder::incRefCount", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_logger_finder_find_logger_deferred, "ILoggerFinder::findLogger", "borrowed logger discovery and logger lifetime require a dedicated safe managed bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_profiler_report_layer_time_deferred, "IProfiler::reportLayerTime", "profiler callback into user code requires an application-owned managed callback bridge")

JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_runtime_get_logger_deferred, "IRuntime::getLogger", "borrowed logger ownership is not safe across the C ABI without a dedicated handle model")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_runtime_get_plugin_registry_deferred, "IRuntime::getPluginRegistry", "borrowed plugin registry ownership is not safe across the C ABI without a dedicated snapshot model")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_runtime_load_runtime_deferred, "IRuntime::loadRuntime", "runtime plugin library loading and returned runtime lifetime require a dedicated safe bridge")

JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_caffe_parser_deferred, "Global::createCaffeParser", "legacy Caffe parser ownership and dependency lifetime are not modeled by the deployment bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_consistency_checker_internal_deferred, "Global::createConsistencyChecker_INTERNAL", "internal consistency checker ownership is not modeled by the public bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_nv_caffe_parser_internal_deferred, "Global::createNvCaffeParser_INTERNAL", "legacy Caffe parser internal factory ownership is not modeled by the deployment bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_nv_uff_parser_internal_deferred, "Global::createNvUffParser_INTERNAL", "legacy UFF parser internal factory ownership is not modeled by the deployment bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_onnx_config_deferred, "Global::createONNXConfig", "ONNX config ownership requires a dedicated safe handle before promotion")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_create_uff_parser_deferred, "Global::createUffParser", "legacy UFF parser ownership and dependency lifetime are not modeled by the deployment bridge")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_get_builder_plugin_registry_deferred, "Global::getBuilderPluginRegistry", "borrowed builder plugin registry ownership is not safe across the C ABI without a dedicated snapshot model")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_get_plugin_registry_deferred, "Global::getPluginRegistry", "borrowed global plugin registry ownership is not safe across the C ABI without a dedicated snapshot model")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_get_safe_plugin_registry_deferred, "Global::getSafePluginRegistry", "borrowed safe plugin registry ownership is not safe across the C ABI without a dedicated snapshot model")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_init_lib_nvinfer_plugins_deferred, "Global::initLibNvInferPlugins", "plugin registration mutates global process state and requires explicit managed policy")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_reorder_sub_buffers_deferred, "Global::reorderSubBuffers", "legacy NvUtils raw-buffer transform requires explicit memory ownership and shape validation")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_reshape_weights_deferred, "Global::reshapeWeights", "legacy NvUtils weights transform requires explicit memory ownership and shape validation")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_shutdown_protobuf_library_deferred, "Global::shutdownProtobufLibrary", "process-global protobuf shutdown is unsafe to expose without lifecycle policy")
JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB(jyppx_trt8_global_transpose_sub_buffers_deferred, "Global::transposeSubBuffers", "legacy NvUtils raw-buffer transform requires explicit memory ownership and shape validation")

#undef JYPPX_TRT8_SEVENTH_BATCH_DEFERRED_STUB
