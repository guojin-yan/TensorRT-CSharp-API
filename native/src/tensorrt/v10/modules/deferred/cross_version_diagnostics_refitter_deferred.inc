#define JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(function_name, interface_name, reason) \
    JYPPX_StatusCode function_name(void) \
    { \
        return jyppx::tensorrt::report_not_implemented(kLine, interface_name " deferred: " reason); \
    }

JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_dec_ref_count_deferred, "IErrorRecorder::decRefCount", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_enum_max_deferred, "IErrorRecorder::EnumMax", "error-recorder enum boundary is covered as deferred metadata until recorder ownership is modeled")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_get_error_code_deferred, "IErrorRecorder::getErrorCode", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_get_error_desc_deferred, "IErrorRecorder::getErrorDesc", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_get_interface_info_deferred, "IErrorRecorder::getInterfaceInfo", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_get_nb_errors_deferred, "IErrorRecorder::getNbErrors", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_has_overflowed_deferred, "IErrorRecorder::hasOverflowed", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_error_recorder_inc_ref_count_deferred, "IErrorRecorder::incRefCount", "application-owned recorder callback lifetime requires a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_logger_finder_find_logger_deferred, "ILoggerFinder::findLogger", "borrowed logger discovery and logger lifetime require a dedicated safe managed bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_profiler_report_layer_time_deferred, "IProfiler::reportLayerTime", "profiler callback into user code requires an application-owned managed callback bridge")
JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB(jyppx_trt10_refitter_set_named_weights_with_location_deferred, "IRefitter::setNamedWeightsWithLocation", "weights location and memory ownership require a dedicated safe managed bridge")

#undef JYPPX_TRT10_DIAGNOSTICS_REFITTER_DEFERRED_STUB
