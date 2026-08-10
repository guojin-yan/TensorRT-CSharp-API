using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class NativeVendorBoundaryGuardTests
{
    [Fact]
    public void TensorRtRuntimeAndBuilderCreationUseNativeExceptionGuards()
    {
        string commonHeader = ReadSource("native", "src", "tensorrt", "common", "object.hpp");
        string commonSource = ReadSource("native", "src", "tensorrt", "common", "object.cpp");

        Assert.Contains("report_vendor_exception", commonHeader);
        Assert.Contains("report_vendor_seh_exception", commonHeader);
        Assert.Contains("capture_vendor_seh_exception_code", commonHeader);
        Assert.Contains("raised a native exception", commonSource);
        Assert.Contains("raised a structured exception with code", commonSource);
        Assert.Contains("return JYPPX_STATUS_RUNTIME_ERROR;", commonSource);

        foreach (string lineDirectory in new[] { "v8", "v10", "v11" })
        {
            string source = ReadSource("native", "src", "tensorrt", lineDirectory, "api.cpp");

            Assert.Contains("JYPPX_StatusCode create_infer_runtime_with_seh_guard", source);
            Assert.Contains("JYPPX_StatusCode create_infer_builder_with_seh_guard", source);
            Assert.Contains("JYPPX_StatusCode create_infer_runtime_with_guard", source);
            Assert.Contains("JYPPX_StatusCode create_infer_builder_with_guard", source);
            Assert.Contains("__try", source);
            Assert.Contains("__except (jyppx::tensorrt::capture_vendor_seh_exception_code", source);
            Assert.Contains("report_vendor_seh_exception(kLine, \"runtime creation\"", source);
            Assert.Contains("report_vendor_seh_exception(kLine, \"builder creation\"", source);
            Assert.Contains("catch (const std::exception& exception)", source);
            Assert.Contains("catch (...)", source);
            Assert.Contains("report_vendor_exception(kLine, \"runtime creation\"", source);
            Assert.Contains("report_vendor_exception(kLine, \"builder creation\"", source);

            string runtimeCreate = ExtractBetween(
                source,
                $"JYPPX_StatusCode jyppx_trt{lineDirectory[1..]}_runtime_create",
                $"JYPPX_StatusCode jyppx_trt{lineDirectory[1..]}_builder_create");
            Assert.Contains("create_infer_runtime_with_guard(*logger_payload, &runtime);", runtimeCreate);
            Assert.DoesNotContain("nvinfer1::createInferRuntime(*logger_payload)", runtimeCreate);

            string builderCreate = ExtractBetween(
                source,
                $"JYPPX_StatusCode jyppx_trt{lineDirectory[1..]}_builder_create",
                $"JYPPX_StatusCode jyppx_trt{lineDirectory[1..]}_builder_platform_has_fast_fp16");
            Assert.Contains("create_infer_builder_with_guard(*logger_payload, &builder);", builderCreate);
            Assert.DoesNotContain("nvinfer1::createInferBuilder(*logger_payload)", builderCreate);
        }
    }

    [Fact]
    public void TensorRt8OnnxParserCreationConvertsVendorExceptionsBeforeCreatingOwnerHandle()
    {
        string source = ReadSource("native", "src", "tensorrt", "v8", "modules", "parser", "parser_inspector.inc");

        Assert.Contains("JYPPX_StatusCode create_onnx_parser_with_seh_guard", source);
        Assert.Contains("JYPPX_StatusCode create_onnx_parser_with_guard", source);
        Assert.Contains("__except (jyppx::tensorrt::capture_vendor_seh_exception_code", source);
        Assert.Contains("report_vendor_seh_exception(kLine, \"ONNX parser creation\"", source);
        Assert.Contains("catch (const std::exception& exception)", source);
        Assert.Contains("report_vendor_exception(kLine, \"ONNX parser creation\"", source);
        Assert.Contains("*out_parser = nullptr;", source);
        Assert.Contains("status = create_onnx_parser_with_guard(*network_payload, *logger_payload, &parser);", source);
        Assert.DoesNotContain("ONNX parser creation is disabled on Windows", source);

        string publicCreate = ExtractBetween(
            source,
            "JYPPX_StatusCode jyppx_trt8_onnx_parser_create",
            "JYPPX_StatusCode jyppx_trt8_onnx_parser_parse_from_file");
        Assert.Contains("create_onnx_parser_with_guard", publicCreate);
        Assert.Contains("create_handle_with_payload", publicCreate);
        Assert.DoesNotContain("nvonnxparser::createParser(*network_payload, *logger_payload)", publicCreate);
    }

    [Fact]
    public void GlobalPluginRegistryReadOnlyProbeConvertsNativeExceptionsToStatusCodes()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");

        Assert.Contains("jyppx_trt_report_readonly_probe_exception", source);
        Assert.Contains("jyppx::tensorrt::report_vendor_exception", source);
        Assert.Contains("jyppx::tensorrt::report_vendor_seh_exception", source);
        Assert.Contains("__try", source);
        Assert.Contains("__except (jyppx::tensorrt::capture_vendor_seh_exception_code", source);
        Assert.Contains("unknown native exception", source);
        Assert.Contains("*out_registry = ::getPluginRegistry();", source);
        Assert.Contains("*out_registry = nvinfer1::getBuilderPluginRegistry", source);
        Assert.Contains("auto* registry = nvinfer1::getBuilderSafePluginRegistry", source);
        Assert.Contains("*out_creators = registry->getAllCreators(out_count);", source);
        Assert.Contains("*out_creator = registry->getCreator(plugin_name, plugin_version, plugin_namespace);", source);
        Assert.Contains("registry->getAllCreators(out_count);", source);
        Assert.Contains("registry->getAllCreatorsRecursive(out_count);", source);

        AssertSehCallIsGuarded(source, "*out_registry = ::getPluginRegistry();", "global plugin registry query");
        AssertSehCallIsGuarded(source, "*out_registry = nvinfer1::getBuilderPluginRegistry", "builder capability plugin registry query");
        AssertSehCallIsGuarded(source, "auto* registry = nvinfer1::getBuilderSafePluginRegistry", "builder safe plugin registry existence query");
        AssertSehCallIsGuarded(source, "*out_creators = registry->getAllCreators(out_count);", "feature_name");
        AssertSehCallIsGuarded(source, "registry->getAllCreatorsRecursive(out_count);", "feature_name");
        AssertSehCallIsGuarded(source, "*out_creator = registry->getCreator(plugin_name, plugin_version, plugin_namespace);", "feature_name");
    }

    [Fact]
    public void PluginRegistryObjectReadOnlyCallsUseGuardedNativeHelpers()
    {
        string pluginInventory = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string globalProbe = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");

        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_builder_plugin_registry_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_inventory_registry_creators_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_inventory_registry_recursive_creator_count_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_plugin_registry_has_error_recorder_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_plugin_registry_parent_search_enabled_with_seh_guard", pluginInventory);

        AssertSehCallIsGuarded(pluginInventory, "*out_registry = &builder_payload->getPluginRegistry();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_creators = registry->getAllCreators(out_count);", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "registry->getAllCreatorsRecursive(out_count);", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");

        string builderRegistryExports = ExtractBetween(
            pluginInventory,
            "JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_get_creator_count)",
            "JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_name)");
        Assert.Contains("jyppx_trt_get_inventory_registry_creator_count(", builderRegistryExports);
        Assert.Contains("jyppx_trt_get_inventory_registry_recursive_creator_count(", builderRegistryExports);
        Assert.Contains("jyppx_trt_get_plugin_registry_has_error_recorder(", builderRegistryExports);
        Assert.Contains("jyppx_trt_get_plugin_registry_parent_search_enabled(", builderRegistryExports);
        Assert.DoesNotContain("registry->getAllCreators(", builderRegistryExports);
        Assert.DoesNotContain("registry->getAllCreatorsRecursive(", builderRegistryExports);
        Assert.DoesNotContain("registry->getErrorRecorder()", builderRegistryExports);
        Assert.DoesNotContain("registry->isParentSearchEnabled()", builderRegistryExports);

        string runtimeRegistryExports = ExtractBetween(
            pluginInventory,
            "JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(runtime_plugin_registry_get_creator_count)",
            "JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(runtime_plugin_registry_has_error_recorder)");
        Assert.Contains("jyppx_trt_get_inventory_registry_creator_count(", runtimeRegistryExports);
        Assert.Contains("jyppx_trt_get_inventory_registry_recursive_creator_count(", runtimeRegistryExports);
        Assert.DoesNotContain("registry->getAllCreators(", runtimeRegistryExports);
        Assert.DoesNotContain("registry->getAllCreatorsRecursive(", runtimeRegistryExports);

        string builderCreatorLookup = ExtractBetween(
            pluginInventory,
            "JYPPX_StatusCode jyppx_trt_get_plugin_creator",
            "JYPPX_StatusCode jyppx_trt_report_plugin_inventory_exception");
        Assert.Contains("jyppx_trt_get_inventory_registry_creators(", builderCreatorLookup);
        Assert.DoesNotContain("registry->getAllCreators(", builderCreatorLookup);

        string globalRegistryPropertyExports = ExtractBetween(
            globalProbe,
            "JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_registry_has_error_recorder)",
            "JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_name)");
        Assert.Contains("jyppx_trt_get_plugin_registry_has_error_recorder(", globalRegistryPropertyExports);
        Assert.Contains("jyppx_trt_get_plugin_registry_parent_search_enabled(", globalRegistryPropertyExports);
        Assert.DoesNotContain("registry->getErrorRecorder()", globalRegistryPropertyExports);
        Assert.DoesNotContain("registry->isParentSearchEnabled()", globalRegistryPropertyExports);

        string builderCapabilityRegistryPropertyExports = ExtractBetween(
            globalProbe,
            "JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_has_error_recorder)",
            "JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_name)");
        Assert.Contains("jyppx_trt_get_plugin_registry_has_error_recorder(", builderCapabilityRegistryPropertyExports);
        Assert.Contains("jyppx_trt_get_plugin_registry_parent_search_enabled(", builderCapabilityRegistryPropertyExports);
        Assert.DoesNotContain("registry->getErrorRecorder()", builderCapabilityRegistryPropertyExports);
        Assert.DoesNotContain("registry->isParentSearchEnabled()", builderCapabilityRegistryPropertyExports);
    }

    [Fact]
    public void CallbackAllocatorSafeControlsUseGuardedNativeHelpers()
    {
        string runtimeControls = ReadSource("native", "src", "tensorrt", "common", "runtime_controls.inc");
        string allocatorControls = ReadSource("native", "src", "tensorrt", "common", "execution_context_allocator_controls.inc");
        string errorRecorderBoundaryControls = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string refitterControls = ReadSource("native", "src", "tensorrt", "common", "refitter_controls.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt8Manifest = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-engine-network-context-error-recorder-controls.manifest.json");
        string trt10Manifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-engine-network-context-error-recorder-controls.manifest.json");
        string trt11Diagnostics = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "diagnostics.inc");
        string trt11BoundaryControls = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string boundaryInterop =
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Builder", "NativeBridgeApi.BuilderBoundaryControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Engine", "NativeBridgeApi.EngineBoundaryControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Execution", "NativeBridgeApi.ExecutionContextBoundaryControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Network", "NativeBridgeApi.NetworkBoundaryControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OwnerErrorRecorderSnapshotShared.cs");
        string publicBoundaryWrappers = string.Join(
            Environment.NewLine,
            ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11BoundaryControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Network", "TensorRtNetworkDefinition.Trt11BoundaryControls.cs"));

        Assert.Contains("runtime_has_error_recorder_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_get_error_recorder_snapshot_info_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_get_error_recorder_error_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_clear_error_recorder_with_seh_guard", runtimeControls);
        Assert.Contains("runtime_clear_gpu_allocator_with_seh_guard", runtimeControls);
        Assert.Contains("execution_context_has_output_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_has_temporary_storage_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_clear_output_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("execution_context_clear_temporary_storage_allocator_with_seh_guard", allocatorControls);
        Assert.Contains("context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE", allocatorControls);
        Assert.Contains("context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", allocatorControls);
        Assert.Contains("context_payload->setOutputAllocator(tensor_name, nullptr)", allocatorControls);
        Assert.Contains("context_payload->setTemporaryStorageAllocator(nullptr)", allocatorControls);
        Assert.Contains("#include \"../common/execution_context_allocator_controls.inc\"", trt8Api);
        Assert.Contains("#include \"../common/execution_context_allocator_controls.inc\"", trt10Api);
        Assert.Contains("#include \"../common/error_recorder_boundary_controls.inc\"", trt8Api);
        Assert.Contains("#include \"../common/error_recorder_boundary_controls.inc\"", trt10Api);
        AssertSehCallIsGuarded(runtimeControls, "*out_has_recorder = runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "auto* recorder = runtime_payload->getErrorRecorder();", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "out_info->error_count = error_count > 0 ? error_count : 0;", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "runtime_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(runtimeControls, "runtime_payload->setGpuAllocator(nullptr);", "feature_name");

        string runtimeExports = ExtractBetween(
            runtimeControls,
            "JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_has_error_recorder)",
            "***END***",
            allowEndOfText: true);
        Assert.Contains("runtime_has_error_recorder_with_seh_guard(", runtimeExports);
        Assert.Contains("runtime_get_error_recorder_snapshot_info_with_seh_guard(", runtimeExports);
        Assert.Contains("runtime_get_error_recorder_error_with_seh_guard(", runtimeExports);
        Assert.Contains("runtime_clear_error_recorder_with_seh_guard(", runtimeExports);
        Assert.Contains("runtime_clear_gpu_allocator_with_seh_guard(", runtimeExports);
        Assert.DoesNotContain("runtime_payload->getErrorRecorder()", runtimeExports);
        Assert.DoesNotContain("recorder->getNbErrors()", runtimeExports);
        Assert.DoesNotContain("recorder->getErrorDesc(index)", runtimeExports);
        Assert.DoesNotContain("runtime_payload->setErrorRecorder(nullptr)", runtimeExports);
        Assert.DoesNotContain("runtime_payload->setGpuAllocator(nullptr)", runtimeExports);

        foreach (string manifest in new[] { trt8Manifest, trt10Manifest })
        {
            Assert.Contains("engine-has-error-recorder", manifest);
            Assert.Contains("engine-clear-error-recorder", manifest);
            Assert.Contains("engine-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("engine-get-error-recorder-error", manifest);
            Assert.Contains("execution-context-has-error-recorder", manifest);
            Assert.Contains("execution-context-clear-error-recorder", manifest);
            Assert.Contains("execution-context-get-error-recorder-snapshot-info", manifest);
            Assert.Contains("execution-context-get-error-recorder-error", manifest);
            Assert.Contains("network-has-error-recorder", manifest);
            Assert.Contains("network-clear-error-recorder", manifest);
            Assert.Contains("engine-inspector-has-error-recorder", manifest);
            Assert.Contains("engine-inspector-clear-error-recorder", manifest);
            Assert.Contains("JYPPX_Boolean*", manifest);
            Assert.DoesNotContain("IErrorRecorder", manifest);
        }

        Assert.Contains("engine_has_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("engine_clear_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("engine_get_error_recorder_snapshot_info_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("engine_get_error_recorder_error_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("execution_context_has_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("execution_context_clear_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("execution_context_get_error_recorder_snapshot_info_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("execution_context_get_error_recorder_error_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("network_has_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("network_clear_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("engine_inspector_has_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("engine_inspector_clear_error_recorder_with_seh_guard", errorRecorderBoundaryControls);
        Assert.Contains("get_engine_payload_for_error_recorder_boundary", errorRecorderBoundaryControls);
        Assert.Contains("get_context_payload_for_error_recorder_boundary", errorRecorderBoundaryControls);
        Assert.Contains("get_network_payload_for_error_recorder_boundary", errorRecorderBoundaryControls);
        Assert.Contains("get_engine_inspector_payload_for_error_recorder_boundary", errorRecorderBoundaryControls);
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "*out_has_recorder = engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "engine_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "*out_has_recorder = context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "context_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "*out_has_recorder = network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "network_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "*out_has_recorder = inspector_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(errorRecorderBoundaryControls, "inspector_payload->setErrorRecorder(nullptr);", "feature_name");

        string errorRecorderBoundaryExports = ExtractBetween(
            errorRecorderBoundaryControls,
            "JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_has_error_recorder)",
            "***END***",
            allowEndOfText: true);
        Assert.Contains("engine_has_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("engine_clear_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("execution_context_has_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("execution_context_clear_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("network_has_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("network_clear_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("engine_inspector_has_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.Contains("engine_inspector_clear_error_recorder_with_seh_guard(", errorRecorderBoundaryExports);
        Assert.DoesNotContain("engine_payload->getErrorRecorder()", errorRecorderBoundaryExports);
        Assert.DoesNotContain("engine_payload->setErrorRecorder(nullptr)", errorRecorderBoundaryExports);
        Assert.DoesNotContain("context_payload->getErrorRecorder()", errorRecorderBoundaryExports);
        Assert.DoesNotContain("context_payload->setErrorRecorder(nullptr)", errorRecorderBoundaryExports);
        Assert.DoesNotContain("network_payload->getErrorRecorder()", errorRecorderBoundaryExports);
        Assert.DoesNotContain("network_payload->setErrorRecorder(nullptr)", errorRecorderBoundaryExports);
        Assert.DoesNotContain("inspector_payload->getErrorRecorder()", errorRecorderBoundaryExports);
        Assert.DoesNotContain("inspector_payload->setErrorRecorder(nullptr)", errorRecorderBoundaryExports);

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_has_error_recorder", boundaryInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_has_error_recorder", boundaryInterop);
        Assert.Contains("支持 TensorRT 8/10/11", publicBoundaryWrappers);
        Assert.DoesNotContain("public IntPtr", publicBoundaryWrappers);
        Assert.DoesNotContain("public nint", publicBoundaryWrappers);

        Assert.Contains("refitter_has_logger_with_seh_guard", refitterControls);
        Assert.Contains("refitter_has_error_recorder_with_seh_guard", refitterControls);
        Assert.Contains("refitter_clear_error_recorder_with_seh_guard", refitterControls);
        AssertSehCallIsGuarded(refitterControls, "*out_has_logger = payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(refitterControls, "*out_has_recorder = payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(refitterControls, "payload->setErrorRecorder(nullptr);", "feature_name");

        string refitterExports = ExtractBetween(
            refitterControls,
            "JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_has_logger)",
            "JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_set_dynamic_range)");
        Assert.Contains("refitter_has_logger_with_seh_guard(", refitterExports);
        Assert.Contains("refitter_has_error_recorder_with_seh_guard(", refitterExports);
        Assert.Contains("refitter_clear_error_recorder_with_seh_guard(", refitterExports);
        Assert.DoesNotContain("payload->getLogger()", refitterExports);
        Assert.DoesNotContain("payload->getErrorRecorder()", refitterExports);
        Assert.DoesNotContain("payload->setErrorRecorder(nullptr)", refitterExports);

        Assert.Contains("jyppx_trt10_builder_config_set_progress_monitor_with_seh_guard", trt10Api);
        Assert.Contains("jyppx_trt10_builder_config_has_progress_monitor_with_seh_guard", trt10Api);
        Assert.Contains("jyppx_trt10_builder_config_clear_progress_monitor_with_seh_guard", trt10Api);
        AssertSehCallIsGuarded(trt10Api, "config_payload->setProgressMonitor(monitor_payload);", "feature_name");
        AssertSehCallIsGuarded(trt10Api, "*out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt10Api, "config_payload->setProgressMonitor(nullptr);", "feature_name");

        string trt10ProgressMonitorExports = ExtractBetween(
            trt10Api,
            "JYPPX_StatusCode jyppx_trt10_builder_config_set_progress_monitor(JYPPX_TensorRtBuilderConfig* config",
            "JYPPX_StatusCode jyppx_trt10_builder_create_network");
        Assert.Contains("jyppx_trt10_builder_config_set_progress_monitor_with_seh_guard(", trt10ProgressMonitorExports);
        Assert.Contains("jyppx_trt10_builder_config_has_progress_monitor_with_seh_guard(", trt10ProgressMonitorExports);
        Assert.Contains("jyppx_trt10_builder_config_clear_progress_monitor_with_seh_guard(", trt10ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->setProgressMonitor(monitor_payload)", trt10ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->getProgressMonitor()", trt10ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->setProgressMonitor(nullptr)", trt10ProgressMonitorExports);

        Assert.Contains("trt11_builder_config_set_progress_monitor_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_clear_output_allocator_with_seh_guard", trt11Diagnostics);
        Assert.Contains("trt11_execution_context_has_profiler_with_seh_guard", trt11Diagnostics);
        AssertSehCallIsGuarded(trt11Diagnostics, "config_payload->setProgressMonitor(monitor_payload);", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "config_payload->setProgressMonitor(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_has_recorder = inspector_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_has_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "context_payload->setProfiler(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11Diagnostics, "*out_has_profiler = context_payload->getProfiler() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");

        string trt11ProgressMonitorExports = ExtractBetween(
            trt11Diagnostics,
            "JYPPX_StatusCode jyppx_trt11_builder_config_set_progress_monitor",
            "JYPPX_StatusCode jyppx_trt11_network_mark_debug");
        Assert.Contains("trt11_builder_config_set_progress_monitor_with_seh_guard(", trt11ProgressMonitorExports);
        Assert.Contains("trt11_builder_config_has_progress_monitor_with_seh_guard(", trt11ProgressMonitorExports);
        Assert.Contains("trt11_builder_config_clear_progress_monitor_with_seh_guard(", trt11ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->setProgressMonitor(monitor_payload)", trt11ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->getProgressMonitor()", trt11ProgressMonitorExports);
        Assert.DoesNotContain("config_payload->setProgressMonitor(nullptr)", trt11ProgressMonitorExports);

        string trt11InspectorExports = ExtractBetween(
            trt11Diagnostics,
            "JYPPX_StatusCode jyppx_trt11_engine_inspector_has_error_recorder",
            "JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_address_value");
        Assert.Contains("trt11_engine_inspector_has_error_recorder_with_seh_guard(", trt11InspectorExports);
        Assert.DoesNotContain("inspector_payload->getErrorRecorder()", trt11InspectorExports);

        string trt11ContextCallbackAllocatorExports = ExtractBetween(
            trt11Diagnostics,
            "JYPPX_StatusCode jyppx_trt11_execution_context_clear_output_allocator",
            "JYPPX_StatusCode jyppx_trt11_execution_context_has_runtime_config");
        Assert.Contains("trt11_execution_context_clear_output_allocator_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.Contains("trt11_execution_context_clear_temporary_storage_allocator_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.Contains("trt11_execution_context_clear_debug_listener_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.Contains("trt11_execution_context_has_debug_listener_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.Contains("trt11_execution_context_clear_profiler_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.Contains("trt11_execution_context_has_profiler_with_seh_guard(", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->setOutputAllocator(tensor_name, nullptr)", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->setTemporaryStorageAllocator(nullptr)", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->setDebugListener(nullptr)", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->getDebugListener()", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->setProfiler(nullptr)", trt11ContextCallbackAllocatorExports);
        Assert.DoesNotContain("context_payload->getProfiler()", trt11ContextCallbackAllocatorExports);

        Assert.Contains("trt11_builder_clear_gpu_allocator_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_builder_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_builder_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_engine_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_engine_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_execution_context_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_execution_context_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_network_has_error_recorder_with_seh_guard", trt11BoundaryControls);
        Assert.Contains("trt11_network_clear_error_recorder_with_seh_guard", trt11BoundaryControls);
        AssertSehCallIsGuarded(trt11BoundaryControls, "builder_payload->setGpuAllocator(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "*out_has_recorder = builder_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "builder_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "*out_has_recorder = engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "engine_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "*out_has_recorder = context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "context_payload->setErrorRecorder(nullptr);", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "*out_has_recorder = network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(trt11BoundaryControls, "network_payload->setErrorRecorder(nullptr);", "feature_name");

        string trt11BuilderBoundaryExports = ExtractBetween(
            trt11BoundaryControls,
            "JYPPX_StatusCode jyppx_trt11_builder_clear_gpu_allocator",
            "JYPPX_StatusCode jyppx_trt11_builder_reset");
        Assert.Contains("trt11_builder_clear_gpu_allocator_with_seh_guard(", trt11BuilderBoundaryExports);
        Assert.Contains("trt11_builder_has_error_recorder_with_seh_guard(", trt11BuilderBoundaryExports);
        Assert.Contains("trt11_builder_clear_error_recorder_with_seh_guard(", trt11BuilderBoundaryExports);
        Assert.DoesNotContain("builder_payload->setGpuAllocator(nullptr)", trt11BuilderBoundaryExports);
        Assert.DoesNotContain("builder_payload->getErrorRecorder()", trt11BuilderBoundaryExports);
        Assert.DoesNotContain("builder_payload->setErrorRecorder(nullptr)", trt11BuilderBoundaryExports);

        string trt11EngineBoundaryExports = ExtractBetween(
            trt11BoundaryControls,
            "JYPPX_StatusCode jyppx_trt11_engine_has_error_recorder",
            "JYPPX_StatusCode jyppx_trt11_engine_get_aliased_input_tensor");
        Assert.Contains("trt11_engine_has_error_recorder_with_seh_guard(", trt11EngineBoundaryExports);
        Assert.Contains("trt11_engine_clear_error_recorder_with_seh_guard(", trt11EngineBoundaryExports);
        Assert.DoesNotContain("engine_payload->getErrorRecorder()", trt11EngineBoundaryExports);
        Assert.DoesNotContain("engine_payload->setErrorRecorder(nullptr)", trt11EngineBoundaryExports);

        string trt11ContextBoundaryExports = ExtractBetween(
            trt11BoundaryControls,
            "JYPPX_StatusCode jyppx_trt11_execution_context_has_error_recorder",
            "JYPPX_StatusCode jyppx_trt11_network_has_error_recorder");
        Assert.Contains("trt11_execution_context_has_error_recorder_with_seh_guard(", trt11ContextBoundaryExports);
        Assert.Contains("trt11_execution_context_clear_error_recorder_with_seh_guard(", trt11ContextBoundaryExports);
        Assert.DoesNotContain("context_payload->getErrorRecorder()", trt11ContextBoundaryExports);
        Assert.DoesNotContain("context_payload->setErrorRecorder(nullptr)", trt11ContextBoundaryExports);

        string trt11NetworkBoundaryExports = ExtractBetween(
            trt11BoundaryControls,
            "JYPPX_StatusCode jyppx_trt11_network_has_error_recorder",
            "JYPPX_StatusCode jyppx_trt11_network_remove_tensor");
        Assert.Contains("trt11_network_has_error_recorder_with_seh_guard(", trt11NetworkBoundaryExports);
        Assert.Contains("trt11_network_clear_error_recorder_with_seh_guard(", trt11NetworkBoundaryExports);
        Assert.DoesNotContain("network_payload->getErrorRecorder()", trt11NetworkBoundaryExports);
        Assert.DoesNotContain("network_payload->setErrorRecorder(nullptr)", trt11NetworkBoundaryExports);
    }

    [Fact]
    public void TensorRt8And10NoVendorFallbackHelpersStayFreeOfTensorRtTypes()
    {
        string trt8Api = NormalizeLineEndings(ReadSource("native", "src", "tensorrt", "v8", "api.cpp"));
        string trt10Api = NormalizeLineEndings(ReadSource("native", "src", "tensorrt", "v10", "api.cpp"));
        string trt8Parser = NormalizeLineEndings(ReadSource("native", "src", "tensorrt", "v8", "modules", "parser", "parser_inspector.inc"));
        string trt10Parser = NormalizeLineEndings(ReadSource("native", "src", "tensorrt", "v10", "modules", "parser", "parser_inspector.inc"));
        string runtimeControls = NormalizeLineEndings(ReadSource("native", "src", "tensorrt", "common", "runtime_controls.inc"));

        string trt8NoVendorFallback = ExtractBetween(
            trt8Api,
            "#if !JYPPX_HAS_TENSORRT",
            "JYPPX_StatusCode jyppx_trt8_query_adapter_info");
        string trt10NoVendorFallback = ExtractBetween(
            trt10Api,
            "#if !JYPPX_HAS_TENSORRT",
            "bool trt10_vendor_available()");

        foreach (string fallback in new[] { trt8NoVendorFallback, trt10NoVendorFallback })
        {
            Assert.Contains("JYPPX_StatusCode validate_c_string", fallback);
            Assert.Contains("JYPPX_StatusCode validate_named_tensor", fallback);
            Assert.Contains("JYPPX_StatusCode validate_index", fallback);
            Assert.Contains("size_t get_data_type_size(const int32_t data_type)", fallback);
            Assert.Contains("switch (data_type)", fallback);
            Assert.DoesNotContain("nvinfer1::", fallback);
            Assert.DoesNotContain("static_cast<nvinfer1::DataType>(data_type)", fallback);
        }

        Assert.Contains("#define JYPPX_TRT8_VALIDATE_NAMED_TENSOR_DEFINED 1", trt8NoVendorFallback);
        Assert.Contains("#define JYPPX_TRT10_VALIDATE_NAMED_TENSOR_DEFINED 1", trt10NoVendorFallback);
        Assert.Contains("#ifndef JYPPX_TRT8_VALIDATE_NAMED_TENSOR_DEFINED", trt8Parser);
        Assert.Contains("#ifndef JYPPX_TRT10_VALIDATE_NAMED_TENSOR_DEFINED", trt10Parser);

        string resetHelpers = ExtractBetween(
            runtimeControls,
            "static void reset_error_recorder_snapshot_info",
            "#if JYPPX_HAS_TENSORRT");
        Assert.Contains("static void reset_error_record_info", resetHelpers);
        Assert.DoesNotContain("nvinfer1::", resetHelpers);

        Assert.Contains("#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10\nclass ManagedProgressMonitor final", trt10Api);
        Assert.Contains("#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10\nJYPPX_StatusCode copy_interface_info_to_buffer", trt10Api);
        Assert.Contains("#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10\n    auto* monitor", trt10Api);
    }

    [Fact]
    public void PluginCreatorMetadataAndFieldsUseGuardedNativeHelpers()
    {
        string pluginInventory = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string globalProbe = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");

        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_creator_name_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_creator_version_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_creator_namespace_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_creator_field_collection_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_creator_interface_info_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_plugin_field_name_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_get_plugin_field_metadata_with_seh_guard", pluginInventory);
        Assert.Contains("JYPPX_StatusCode jyppx_trt_validate_plugin_field_storage", pluginInventory);
        Assert.Contains("returned a non-empty PluginFieldCollection without field storage", pluginInventory);

        AssertSehCallIsGuarded(pluginInventory, "*out_fields = legacy_creator->getFieldNames();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_fields = v3_creator->getFieldNames();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = legacy_creator->getPluginName();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = v3_creator->getPluginName();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = legacy_creator->getPluginVersion();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = v3_creator->getPluginVersion();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = legacy_creator->getPluginNamespace();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_value = v3_creator->getPluginNamespace();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_info = creator->getInterfaceInfo();", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "*out_name = fields->fields[field_index].name;", "feature_name");
        AssertSehCallIsGuarded(pluginInventory, "const auto& field = fields->fields[field_index];", "feature_name");

        string builderOwnedExports = ExtractBetween(
            pluginInventory,
            "JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_name)",
            "#undef JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY");
        Assert.Contains("jyppx_trt_get_creator_name(creator, &name", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_creator_version(creator, &version", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_creator_namespace(creator, &plugin_namespace", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_creator_interface_info(creator, &info", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_creator_field_collection(creator, &fields", builderOwnedExports);
        Assert.Contains("jyppx_trt_validate_plugin_field_storage(fields, field_count", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_plugin_field_name(fields, field_index, &field_name", builderOwnedExports);
        Assert.Contains("jyppx_trt_get_plugin_field_metadata(", builderOwnedExports);
        Assert.DoesNotContain("->getPluginName()", builderOwnedExports);
        Assert.DoesNotContain("->getPluginVersion()", builderOwnedExports);
        Assert.DoesNotContain("->getPluginNamespace()", builderOwnedExports);
        Assert.DoesNotContain("->getFieldNames()", builderOwnedExports);
        Assert.DoesNotContain("->getInterfaceInfo()", builderOwnedExports);
        Assert.DoesNotContain("fields->fields[field_index]", builderOwnedExports);

        string globalExports = ExtractBetween(
            globalProbe,
            "JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_name)",
            "#undef JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE");
        Assert.Contains("jyppx_trt_get_creator_name(creator, &name", globalExports);
        Assert.Contains("jyppx_trt_get_creator_version(creator, &version", globalExports);
        Assert.Contains("jyppx_trt_get_creator_namespace(creator, &plugin_namespace", globalExports);
        Assert.Contains("jyppx_trt_get_creator_interface_info(creator, &info", globalExports);
        Assert.Contains("jyppx_trt_get_creator_field_collection(creator, &fields", globalExports);
        Assert.Contains("jyppx_trt_validate_plugin_field_storage(fields, field_count", globalExports);
        Assert.Contains("jyppx_trt_get_plugin_field_name(fields, field_index, &field_name", globalExports);
        Assert.Contains("jyppx_trt_get_plugin_field_metadata(", globalExports);
        Assert.DoesNotContain("->getPluginName()", globalExports);
        Assert.DoesNotContain("->getPluginVersion()", globalExports);
        Assert.DoesNotContain("->getPluginNamespace()", globalExports);
        Assert.DoesNotContain("->getFieldNames()", globalExports);
        Assert.DoesNotContain("->getInterfaceInfo()", globalExports);
        Assert.DoesNotContain("fields->fields[field_index]", globalExports);
    }

    [Fact]
    public void OnnxParserSupportReadOnlyCallsUseGuardedNativeHelpers()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "onnx_parser_support.inc");

        Assert.Contains("get_onnx_parser_used_vc_plugin_libraries_with_seh_guard", source);
        Assert.Contains("onnx_parser_supports_model_v2_with_seh_guard", source);
        Assert.Contains("get_onnx_parser_subgraph_count_with_seh_guard", source);
        Assert.Contains("is_onnx_parser_subgraph_supported_with_seh_guard", source);
        Assert.Contains("get_onnx_parser_subgraph_nodes_with_seh_guard", source);
        Assert.Contains("onnx_parser_layer_output_tensor_exists_with_seh_guard", source);

        AssertSehCallIsGuarded(source, "*out_libraries = parser->getUsedVCPluginLibraries(*out_count);", "feature_name");
        AssertSehCallIsGuarded(source, "*out_supported = parser->supportsModelV2(model_data, model_size, model_path) ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(source, "*out_count = parser->getNbSubgraphs();", "feature_name");
        AssertSehCallIsGuarded(source, "*out_supported = parser->isSubgraphSupported(index) ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");
        AssertSehCallIsGuarded(source, "*out_nodes = parser->getSubgraphNodes(subgraph_index, *out_count);", "feature_name");
        AssertSehCallIsGuarded(source, "*out_exists = parser->getLayerOutputTensor(layer_name, output_index) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;", "feature_name");

        string exports = ExtractBetween(
            source,
            "JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_used_vc_plugin_library_count)",
            "#undef JYPPX_TRT_ONNX_PARSER_API");

        Assert.Contains("get_onnx_parser_used_vc_plugin_libraries_with_seh_guard(", exports);
        Assert.Contains("onnx_parser_supports_model_v2_with_seh_guard(", exports);
        Assert.Contains("get_onnx_parser_subgraph_count_with_seh_guard(", exports);
        Assert.Contains("is_onnx_parser_subgraph_supported_with_seh_guard(", exports);
        Assert.Contains("get_onnx_parser_subgraph_nodes_for_support(", exports);
        Assert.Contains("onnx_parser_layer_output_tensor_exists_with_seh_guard(", exports);
        Assert.Contains("count_onnx_parser_subgraphs_by_support(", exports);

        Assert.DoesNotContain("parser_payload->getUsedVCPluginLibraries(", exports);
        Assert.DoesNotContain("parser_payload->supportsModelV2(", exports);
        Assert.DoesNotContain("parser_payload->getNbSubgraphs()", exports);
        Assert.DoesNotContain("parser_payload->isSubgraphSupported(", exports);
        Assert.DoesNotContain("parser_payload->getSubgraphNodes(", exports);
        Assert.DoesNotContain("parser_payload->getLayerOutputTensor(", exports);
    }

    private static void AssertSehCallIsGuarded(string source, string callToken, string diagnosticToken)
    {
        int call = source.IndexOf(callToken, StringComparison.Ordinal);
        Assert.True(call >= 0, $"Unable to find guarded call token: {callToken}");

        int guardStart = source.LastIndexOf("__try", call, StringComparison.Ordinal);
        int guardEnd = source.IndexOf("__except", call, StringComparison.Ordinal);

        Assert.True(guardStart >= 0 && guardEnd > call, $"Call is not enclosed by a native SEH guard: {callToken}");
        string guardedBlock = source[guardStart..Math.Min(source.Length, guardEnd + 512)];
        Assert.Contains(diagnosticToken, guardedBlock);
        Assert.Contains("report_vendor_seh_exception", guardedBlock);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static string NormalizeLineEndings(string text)
    {
        return text.Replace("\r\n", "\n", StringComparison.Ordinal);
    }

    private static string ExtractBetween(string text, string startToken, string endToken, bool allowEndOfText = false)
    {
        int start = text.IndexOf(startToken, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Unable to find start token: {startToken}");

        int end = text.IndexOf(endToken, start, StringComparison.Ordinal);
        if (allowEndOfText && end < 0)
        {
            end = text.Length;
        }

        Assert.True(end > start, $"Unable to find end token: {endToken}");

        return text[start..end];
    }
}
