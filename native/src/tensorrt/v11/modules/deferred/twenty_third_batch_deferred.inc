JYPPX_StatusCode jyppx_trt11_execution_context_execute_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::executeV2 deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_execution_context_no_copy_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::INoCopy deferred: lifetime is governed by bridge SafeHandle/common destroy semantics rather than per-interface vendor destroy calls");
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_communicator_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setCommunicator deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_device_memory_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExecutionContext::setDeviceMemoryV2 deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_build_serialized_network_to_stream_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::buildSerializedNetworkToStream deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_avg_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getAvgTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_builder_optimization_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getBuilderOptimizationLevel deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_nb_plugins_to_serialize_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::getNbPluginsToSerialize deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_avg_timing_iterations_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setAvgTimingIterations deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_builder_optimization_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilderConfig::setBuilderOptimizationLevel deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_builder_get_logger_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getLogger deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt11_builder_get_plugin_registry_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IBuilder::getPluginRegistry deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_expr_builder_constant_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExprBuilder::constant deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_expr_builder_declare_size_tensor_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExprBuilder::declareSizeTensor deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_expr_builder_operation_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IExprBuilder::operation deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_add_verbosity_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::addVerbosity deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_full_text_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getFullTextFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_model_dtype_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getModelDtype deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_model_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getModelFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_print_layer_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getPrintLayerInfo deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_text_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getTextFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_get_verbosity_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::getVerbosityLevel deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_reduce_verbosity_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::reduceVerbosity deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_full_text_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setFullTextFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_model_dtype_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setModelDtype deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_model_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setModelFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_print_layer_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setPrintLayerInfo deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_text_file_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setTextFileName deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_set_verbosity_level_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOnnxConfig::setVerbosityLevel deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_get_layer_output_tensor_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::getLayerOutputTensor deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_get_nb_subgraphs_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::getNbSubgraphs deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_get_subgraph_nodes_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::getSubgraphNodes deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_get_used_vc_plugin_libraries_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::getUsedVCPluginLibraries deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_is_subgraph_supported_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::isSubgraphSupported deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_load_initializer_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::loadInitializer deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_load_model_proto_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::loadModelProto deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_parse_model_proto_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::parseModelProto deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_create_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "createNvOnnxParserRefitter_INTERNAL deferred: IParserRefitter object creation remains blocked until the parser-refitter model-buffer and diagnostic lifetime surface is reviewed");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_clear_errors_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::clearErrors deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_get_error_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::getError deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_get_nb_errors_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::getNbErrors deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_load_initializer_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::loadInitializer deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_load_model_proto_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::loadModelProto deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_refit_from_bytes_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::refitFromBytes deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_refit_from_file_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::refitFromFile deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_refitter_refit_model_proto_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParserRefitter::refitModelProto deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_set_builder_config_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::setBuilderConfig deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_parser_supports_model_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IParser::supportsModelV2 deferred: advanced ONNX parser/model-proto surface needs model-buffer ownership and diagnostic lifetime review");
}

JYPPX_StatusCode jyppx_trt11_onnx_config_create_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "createONNXConfig deferred: legacy ONNX config object and file/path configuration surface are intentionally not lifted into managed ownership");
}

JYPPX_StatusCode jyppx_trt11_global_init_lib_nvinfer_plugins_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "initLibNvInferPlugins deferred: process-wide plugin registration side effects are not exposed by the safe managed bridge");
}

JYPPX_StatusCode jyppx_trt11_global_set_internal_library_path_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "setInternalLibraryPath deferred: process-wide TensorRT internal library path mutation is not safe for library-style managed use");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_dec_ref_count_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::decRefCount deferred: error recorder is an application callback object with ref-count ownership across the TensorRT boundary");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_enum_max_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::EnumMax deferred: coverage artifact from callback interface enum surface, not a callable TensorRT-owned operation");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_get_error_code_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::getErrorCode deferred: error recorder diagnostics require a managed callback/snapshot lifetime design before public binding");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_get_error_desc_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::getErrorDesc deferred: returned diagnostic strings are owned by an application callback recorder and need lifetime review");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::getInterfaceInfo deferred: callback interface metadata is not exposed without a stable managed callback object model");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_get_nb_errors_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::getNbErrors deferred: error recorder diagnostics require a managed callback/snapshot lifetime design before public binding");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_has_overflowed_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::hasOverflowed deferred: error recorder diagnostics require a managed callback/snapshot lifetime design before public binding");
}

JYPPX_StatusCode jyppx_trt11_error_recorder_inc_ref_count_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IErrorRecorder::incRefCount deferred: error recorder is an application callback object with ref-count ownership across the TensorRT boundary");
}

JYPPX_StatusCode jyppx_trt11_logger_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ILogger::getInterfaceInfo deferred: logger is a callback interface and the bridge only exposes safe logger construction/usage, not raw interface metadata");
}

JYPPX_StatusCode jyppx_trt11_logger_finder_find_logger_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ILoggerFinder::findLogger deferred: logger finder returns a borrowed logger callback pointer and needs a managed callback ownership design");
}

JYPPX_StatusCode jyppx_trt11_logger_finder_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "ILoggerFinder::getInterfaceInfo deferred: logger finder is a callback provider interface, not a TensorRT-owned query object in the bridge");
}

JYPPX_StatusCode jyppx_trt11_profiler_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProfiler::getInterfaceInfo deferred: profiler is an application callback interface and needs managed callback ownership before public binding");
}

JYPPX_StatusCode jyppx_trt11_profiler_report_layer_time_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProfiler::reportLayerTime deferred: profiler callbacks are invoked by TensorRT into user code and are not safe as standalone bridge calls");
}

JYPPX_StatusCode jyppx_trt11_debug_listener_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IDebugListener::getInterfaceInfo deferred: debug listener is an application callback interface without a managed callback lifetime model");
}

JYPPX_StatusCode jyppx_trt11_debug_listener_process_debug_tensor_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IDebugListener::processDebugTensor deferred: debug tensor callbacks expose TensorRT-owned memory during enqueue and need callback/lifetime design");
}

JYPPX_StatusCode jyppx_trt11_dimension_expr_get_constant_value_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IDimensionExpr::getConstantValue deferred: dimension expressions are only safely owned inside IExprBuilder/plugin shape callbacks, which remain deferred");
}

JYPPX_StatusCode jyppx_trt11_dimension_expr_is_constant_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IDimensionExpr::isConstant deferred: dimension expressions are only safely owned inside IExprBuilder/plugin shape callbacks, which remain deferred");
}

JYPPX_StatusCode jyppx_trt11_dimension_expr_is_size_tensor_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IDimensionExpr::isSizeTensor deferred: dimension expressions are only safely owned inside IExprBuilder/plugin shape callbacks, which remain deferred");
}

JYPPX_StatusCode jyppx_trt11_gpu_allocator_allocate_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAllocator::allocate deferred: allocator callbacks transfer device-memory ownership and require a dedicated managed allocator contract");
}

JYPPX_StatusCode jyppx_trt11_gpu_allocator_deallocate_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAllocator::deallocate deferred: allocator callbacks transfer device-memory ownership and require a dedicated managed allocator contract");
}

JYPPX_StatusCode jyppx_trt11_gpu_allocator_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAllocator::getInterfaceInfo deferred: allocator is an application callback object, not a TensorRT-owned query object in the bridge");
}

JYPPX_StatusCode jyppx_trt11_gpu_allocator_reallocate_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAllocator::reallocate deferred: allocator callbacks transfer device-memory ownership and require a dedicated managed allocator contract");
}

JYPPX_StatusCode jyppx_trt11_gpu_async_allocator_allocate_async_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAsyncAllocator::allocateAsync deferred: async allocator callbacks require stream-aware device-memory ownership and thread-safety contracts");
}

JYPPX_StatusCode jyppx_trt11_gpu_async_allocator_deallocate_async_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAsyncAllocator::deallocateAsync deferred: async allocator callbacks require stream-aware device-memory ownership and thread-safety contracts");
}

JYPPX_StatusCode jyppx_trt11_gpu_async_allocator_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IGpuAsyncAllocator::getInterfaceInfo deferred: async allocator is an application callback object, not a TensorRT-owned query object in the bridge");
}

JYPPX_StatusCode jyppx_trt11_output_allocator_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOutputAllocator::getInterfaceInfo deferred: output allocator is an application callback object with enqueue-time memory ownership");
}

JYPPX_StatusCode jyppx_trt11_output_allocator_notify_shape_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOutputAllocator::notifyShape deferred: output allocator callbacks are invoked during enqueue and need managed callback lifetime design");
}

JYPPX_StatusCode jyppx_trt11_output_allocator_reallocate_output_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IOutputAllocator::reallocateOutput deferred: output allocator callbacks transfer output-memory ownership during enqueue");
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProgressMonitor::getInterfaceInfo deferred: progress monitor is an application callback interface without a managed callback lifetime model");
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_phase_finish_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProgressMonitor::phaseFinish deferred: build progress callbacks are invoked by TensorRT into user code and need callback ownership design");
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_phase_start_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProgressMonitor::phaseStart deferred: build progress callbacks are invoked by TensorRT into user code and need callback ownership design");
}

JYPPX_StatusCode jyppx_trt11_progress_monitor_step_complete_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IProgressMonitor::stepComplete deferred: callback return values can cancel a build and need explicit managed cancellation semantics");
}

JYPPX_StatusCode jyppx_trt11_stream_reader_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamReader::getInterfaceInfo deferred: stream reader is an application-provided deserialization callback interface");
}

JYPPX_StatusCode jyppx_trt11_stream_reader_read_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamReader::read deferred: deserialization stream callbacks require managed stream pinning and buffer lifetime design");
}

JYPPX_StatusCode jyppx_trt11_stream_reader_v2_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamReaderV2::getInterfaceInfo deferred: stream reader v2 is an application-provided deserialization callback interface");
}

JYPPX_StatusCode jyppx_trt11_stream_reader_v2_read_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamReaderV2::read deferred: async deserialization stream callbacks require stream-aware buffer lifetime design");
}

JYPPX_StatusCode jyppx_trt11_stream_reader_v2_seek_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamReaderV2::seek deferred: deserialization stream cursor callbacks require a managed stream bridge contract");
}

JYPPX_StatusCode jyppx_trt11_stream_writer_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamWriter::getInterfaceInfo deferred: stream writer is an application-provided serialization callback interface");
}

JYPPX_StatusCode jyppx_trt11_stream_writer_write_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IStreamWriter::write deferred: serialization stream callbacks require managed stream pinning and buffer lifetime design");
}

JYPPX_StatusCode jyppx_trt11_versioned_interface_get_api_language_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IVersionedInterface::getAPILanguage deferred: base interface metadata is only meaningful through concrete owned objects, not a raw base pointer ABI");
}

JYPPX_StatusCode jyppx_trt11_versioned_interface_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IVersionedInterface::getInterfaceInfo deferred: base interface metadata is only meaningful through concrete owned objects, not a raw base pointer ABI");
}

JYPPX_StatusCode jyppx_trt11_network_add_plugin_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::addPluginV2 deferred: plugin instance pointers and ownership are not exposed by the safe managed bridge");
}

JYPPX_StatusCode jyppx_trt11_network_add_plugin_v3_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::addPluginV3 deferred: plugin instance pointers and ownership are not exposed by the safe managed bridge");
}

JYPPX_StatusCode jyppx_trt11_plugin_v2_layer_get_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV2Layer::getPlugin deferred: returning a borrowed plugin pointer would expose opaque plugin ownership across the C ABI");
}

#define JYPPX_TRT11_PLUGIN_DEFERRED_STUB(function_name, message) \
    JYPPX_StatusCode function_name(void) \
    { \
        return jyppx::tensorrt::report_not_implemented(kLine, message); \
    }

JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_acquire_plugin_resource_deferred, "IPluginRegistry::acquirePluginResource deferred: plugin resources cross registry ownership and require explicit acquire/release lifetime tracking")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_deregister_creator_deferred, "IPluginRegistry::deregisterCreator deferred: registry creator mutation can invalidate borrowed plugin creator pointers")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_deregister_library_deferred, "IPluginRegistry::deregisterLibrary deferred: plugin library unload mutates process-level plugin state and lifetime")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_get_all_creators_recursive_deferred, "IPluginRegistry::getAllCreatorsRecursive deferred: returns borrowed creator pointer arrays beyond the safe inventory snapshot surface")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_load_library_deferred, "IPluginRegistry::loadLibrary deferred: dynamic plugin library loading has process-wide side effects and library-handle lifetime concerns")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_register_creator_deferred, "IPluginRegistry::registerCreator deferred: registering application-owned plugin creators requires callback and ownership design")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_registry_release_plugin_resource_deferred, "IPluginRegistry::releasePluginResource deferred: plugin resource release must be paired with a tracked acquired resource owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_resource_clone_deferred, "IPluginResource::clone deferred: plugin resources are opaque application-provided objects with custom clone/release ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_resource_get_interface_info_deferred, "IPluginResource::getInterfaceInfo deferred: plugin resource metadata belongs to an opaque resource callback object")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_resource_release_deferred, "IPluginResource::release deferred: plugin resource lifetime must be managed through a dedicated resource owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_resource_context_get_error_recorder_deferred, "IPluginResourceContext::getErrorRecorder deferred: resource context returns a borrowed callback object owned by TensorRT/plugin execution")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_resource_context_get_gpu_allocator_deferred, "IPluginResourceContext::getGpuAllocator deferred: resource context returns a borrowed allocator callback with device-memory ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_clone_deferred, "IPluginV2::clone deferred: plugin clone returns a new opaque plugin instance requiring explicit ownership semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_configure_with_format_deferred, "IPluginV2::configureWithFormat deferred: plugin implementation callback must be invoked by TensorRT with version-specific tensor metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_destroy_deferred, "IPluginV2::destroy deferred: raw plugin instance destruction is not exposed outside a safe plugin owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_enqueue_deferred, "IPluginV2::enqueue deferred: plugin enqueue is a TensorRT-to-plugin callback with raw buffers, workspace and stream ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_nb_outputs_deferred, "IPluginV2::getNbOutputs deferred: plugin implementation metadata is only safe through a tracked plugin instance owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_output_dimensions_deferred, "IPluginV2::getOutputDimensions deferred: plugin dimension callback depends on TensorRT-owned input shape arrays")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_plugin_namespace_deferred, "IPluginV2::getPluginNamespace deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_plugin_type_deferred, "IPluginV2::getPluginType deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_plugin_version_deferred, "IPluginV2::getPluginVersion deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_serialization_size_deferred, "IPluginV2::getSerializationSize deferred: plugin serialization requires plugin-owned buffer sizing semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_tensor_rt_version_deferred, "IPluginV2::getTensorRTVersion deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_get_workspace_size_deferred, "IPluginV2::getWorkspaceSize deferred: plugin workspace sizing belongs to TensorRT plugin execution callbacks")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_initialize_deferred, "IPluginV2::initialize deferred: plugin lifecycle callbacks require a tracked plugin instance owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_serialize_deferred, "IPluginV2::serialize deferred: plugin serialization writes to caller-provided buffers with plugin-specific lifetime")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_set_plugin_namespace_deferred, "IPluginV2::setPluginNamespace deferred: mutating raw plugin instance metadata requires plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_supports_format_deferred, "IPluginV2::supportsFormat deferred: plugin format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_terminate_deferred, "IPluginV2::terminate deferred: plugin lifecycle callbacks require a tracked plugin instance owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_clone_deferred, "IPluginV2DynamicExt::clone deferred: plugin clone returns a new opaque plugin instance requiring explicit ownership semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_configure_plugin_deferred, "IPluginV2DynamicExt::configurePlugin deferred: dynamic plugin configuration is a TensorRT-to-plugin callback")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_enqueue_deferred, "IPluginV2DynamicExt::enqueue deferred: dynamic plugin enqueue exposes raw tensor buffers, workspace and stream ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_get_output_dimensions_deferred, "IPluginV2DynamicExt::getOutputDimensions deferred: dynamic plugin shape callback depends on expression builder and input metadata lifetimes")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_get_workspace_size_deferred, "IPluginV2DynamicExt::getWorkspaceSize deferred: plugin workspace sizing belongs to TensorRT plugin execution callbacks")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_return_deferred, "IPluginV2DynamicExt::return deferred: coverage artifact from plugin callback signature parsing, not a standalone safe operation")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_supports_format_deferred, "IPluginV2DynamicExt::supportsFormat deferred: plugin format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_dynamic_ext_supports_format_combination_deferred, "IPluginV2DynamicExt::supportsFormatCombination deferred: plugin format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_ext_clone_deferred, "IPluginV2Ext::clone deferred: plugin clone returns a new opaque plugin instance requiring explicit ownership semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_ext_configure_plugin_deferred, "IPluginV2Ext::configurePlugin deferred: plugin configuration is a TensorRT-to-plugin callback")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_ext_get_output_data_type_deferred, "IPluginV2Ext::getOutputDataType deferred: plugin output type callback depends on TensorRT-owned input metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_ext_get_tensor_rt_version_deferred, "IPluginV2Ext::getTensorRTVersion deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_io_ext_configure_plugin_deferred, "IPluginV2IOExt::configurePlugin deferred: plugin IO configuration is a TensorRT-to-plugin callback")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_io_ext_get_tensor_rt_version_deferred, "IPluginV2IOExt::getTensorRTVersion deferred: raw plugin instance metadata is not exposed without plugin instance ownership")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_io_ext_supports_format_deferred, "IPluginV2IOExt::supportsFormat deferred: plugin IO format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v2_io_ext_supports_format_combination_deferred, "IPluginV2IOExt::supportsFormatCombination deferred: plugin IO format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_clone_deferred, "IPluginV3::clone deferred: plugin clone returns a new opaque V3 plugin instance requiring explicit ownership semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_get_capability_interface_deferred, "IPluginV3::getCapabilityInterface deferred: returns borrowed capability interface pointers with plugin-owned lifetime")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_configure_plugin_deferred, "IPluginV3OneBuild::configurePlugin deferred: V3 build plugin configuration is a TensorRT-to-plugin callback")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_aliased_input_deferred, "IPluginV3OneBuild::getAliasedInput deferred: V3 build callback returns plugin-defined alias metadata tied to TensorRT build context")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_format_combination_limit_deferred, "IPluginV3OneBuild::getFormatCombinationLimit deferred: V3 build format callback belongs to plugin implementation semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_interface_info_deferred, "IPluginV3OneBuild::getInterfaceInfo deferred: V3 build capability metadata is tied to borrowed plugin capability pointers")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_metadata_string_deferred, "IPluginV3OneBuild::getMetadataString deferred: returned plugin metadata string is owned by a plugin capability interface")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_nb_outputs_deferred, "IPluginV3OneBuild::getNbOutputs deferred: V3 build plugin metadata is only safe through a tracked plugin instance owner")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_nb_tactics_deferred, "IPluginV3OneBuild::getNbTactics deferred: V3 tactic callbacks belong to plugin build implementation semantics")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_output_data_types_deferred, "IPluginV3OneBuild::getOutputDataTypes deferred: V3 output type callbacks depend on TensorRT-owned input metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_output_shapes_deferred, "IPluginV3OneBuild::getOutputShapes deferred: V3 shape callbacks depend on expression builder and TensorRT-owned metadata lifetimes")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_timing_cache_id_deferred, "IPluginV3OneBuild::getTimingCacheID deferred: returned timing cache id string is plugin-owned capability metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_valid_tactics_deferred, "IPluginV3OneBuild::getValidTactics deferred: V3 tactic arrays are plugin-owned build callback data")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_get_workspace_size_deferred, "IPluginV3OneBuild::getWorkspaceSize deferred: V3 workspace sizing belongs to TensorRT plugin execution callbacks")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_build_supports_format_combination_deferred, "IPluginV3OneBuild::supportsFormatCombination deferred: V3 format callbacks are invoked by TensorRT with version-specific metadata")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_core_get_interface_info_deferred, "IPluginV3OneCore::getInterfaceInfo deferred: V3 core capability metadata is tied to borrowed plugin capability pointers")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_core_get_plugin_name_deferred, "IPluginV3OneCore::getPluginName deferred: V3 core plugin metadata is only safe through registry/creator inventory snapshots")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_core_get_plugin_namespace_deferred, "IPluginV3OneCore::getPluginNamespace deferred: V3 core plugin metadata is only safe through registry/creator inventory snapshots")
JYPPX_TRT11_PLUGIN_DEFERRED_STUB(jyppx_trt11_plugin_v3_one_core_get_plugin_version_deferred, "IPluginV3OneCore::getPluginVersion deferred: V3 core plugin metadata is only safe through registry/creator inventory snapshots")

#undef JYPPX_TRT11_PLUGIN_DEFERRED_STUB

JYPPX_StatusCode jyppx_trt11_refitter_get_logger_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRefitter::getLogger deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt11_refitter_set_named_weights_with_location_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRefitter::setNamedWeightsWithLocation deferred: signature parity requires per-version review before it can be promoted from coverage boundary to supported API");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_attach_to_context_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::attachToContext deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_enqueue_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::enqueue deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_get_fields_to_serialize_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::getFieldsToSerialize deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::getInterfaceInfo deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_on_shape_change_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::onShapeChange deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_v3_one_runtime_set_tactic_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3OneRuntime::setTactic deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_runtime_deserialize_cuda_engine_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRuntime::deserializeCudaEngineV2 deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_runtime_get_logger_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRuntime::getLogger deferred: callback or borrowed-object ownership is not safe across the C ABI in this batch");
}

JYPPX_StatusCode jyppx_trt11_runtime_get_plugin_registry_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRuntime::getPluginRegistry deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_runtime_load_runtime_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IRuntime::loadRuntime deferred: runtime serialization/plugin-host-code path needs per-line ownership and redistribution review");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_create_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::createPlugin deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_deserialize_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::deserializePlugin deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_get_field_names_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::getFieldNames deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::getInterfaceInfo deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_get_plugin_name_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::getPluginName deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_get_plugin_namespace_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::getPluginNamespace deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_get_plugin_version_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::getPluginVersion deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_set_plugin_namespace_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreator::setPluginNamespace deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_v3_one_create_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreatorV3One::createPlugin deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_v3_one_get_field_names_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreatorV3One::getFieldNames deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}

JYPPX_StatusCode jyppx_trt11_plugin_creator_v3_one_get_interface_info_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginCreatorV3One::getInterfaceInfo deferred: plugin factory, callback, or opaque plugin ownership needs a dedicated safe-handle design before public binding");
}
