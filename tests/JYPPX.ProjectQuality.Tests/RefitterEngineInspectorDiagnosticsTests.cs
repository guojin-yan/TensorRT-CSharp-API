using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RefitterEngineInspectorDiagnosticsTests
{
    [Fact]
    public void TensorRt8And10RefitterControlsUseRealCountCopyAbi()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-thirty-first-batch-refitter-controls.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-thirty-first-batch-refitter-controls.manifest.json");

        foreach (string manifest in new[] { manifest8, manifest10 })
        {
            Assert.Contains("refitter_get_max_threads", manifest);
            Assert.Contains("refitter_has_error_recorder", manifest);
            Assert.Contains("refitter_clear_error_recorder", manifest);
            Assert.Contains("refitter_get_dynamic_range_tensor_count", manifest);
            Assert.Contains("refitter_get_dynamic_range_tensor_entries", manifest);
            Assert.Contains("refitter_get_missing_weights_count", manifest);
            Assert.Contains("refitter_get_all_weights_count", manifest);
            Assert.Contains("refitter_get_missing_weights_entries", manifest);
            Assert.Contains("refitter_get_all_weights_entries", manifest);
            Assert.Contains("\"type\": \"JYPPX_TensorRtRefitEntryInfo*\", \"direction\": \"out\", \"managedType\": \"IntPtr\", \"moduleManagedType\": \"[Out] NativeTensorRtRefitEntryInfo[]\"", manifest);
            Assert.DoesNotContain("_deferred", manifest);
        }
    }

    [Fact]
    public void NativeRefitterControlsCopyNamesAndGuardBorrowedRecorderQueries()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "refitter_controls.inc");

        Assert.Contains("payload->getTensorsWithDynamicRange(0, nullptr)", source);
        Assert.Contains("payload->getMissingWeights(0, nullptr)", source);
        Assert.Contains("payload->getAllWeights(0, nullptr)", source);
        Assert.Contains("std::vector<char const*> names", source);
        Assert.Contains("copy_refitter_names_to_entries(names, output_entries, copy_count);", source);
        Assert.Contains("JYPPX_STATUS_BUFFER_TOO_SMALL", source);
        Assert.Contains("payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE", source);
        Assert.Contains("payload->setErrorRecorder(nullptr);", source);
        Assert.Contains("refitter_has_error_recorder_with_seh_guard", source);
        Assert.Contains("refitter_clear_error_recorder_with_seh_guard", source);
    }

    [Fact]
    public void ManagedRefitterWrapperExposesManagedValuesAndVersionGuardedRoutes()
    {
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitter.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitter.Trt11Controls.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs");

        Assert.Contains("public int MaxThreads", wrapper);
        Assert.Contains("public bool HasErrorRecorder", wrapper);
        Assert.Contains("public IReadOnlyList<string> GetDynamicRangeTensorNames()", wrapper);
        Assert.Contains("public IReadOnlyList<string> GetMissingNamedWeights()", wrapper);
        Assert.Contains("public IReadOnlyList<string> GetAllNamedWeights()", wrapper);
        Assert.Contains("private static IReadOnlyList<string> ConvertNames", wrapper);
        Assert.Contains("DecodeFixedUtf8(nativeEntries[index].LayerName)", wrapper);
        Assert.Contains("不会暴露 recorder 指针", wrapper);
        Assert.Contains("不会接管 recorder 生命周期", wrapper);
        Assert.DoesNotContain("public IntPtr", wrapper);
        Assert.DoesNotContain("public nint", wrapper);

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_max_threads", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_max_threads", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_max_threads", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_dynamic_range_tensor_count", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_dynamic_range_tensor_count", interop);
        Assert.Contains("throw UnsupportedRefitterFeature(nameof(GetRefitterDynamicRangeTensorCount), \"TensorRT 8 and 10\")", interop);
    }

    [Fact]
    public void EngineInspectorDiagnosticsUseCallerBuffersAndKeepUnsafeRecorderPointersDeferred()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-minimal.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-minimal.manifest.json");
        string manifest11Deployment = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-deployment.manifest.json");
        string manifest11Eleventh = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-eleventh-batch.manifest.json");
        string errorRecorderManifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-engine-network-context-error-recorder-controls.manifest.json");
        string errorRecorderManifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-engine-network-context-error-recorder-controls.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string source8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "parser", "parser_inspector.inc");
        string source10 = ReadSource("native", "src", "tensorrt", "v10", "modules", "parser", "parser_inspector.inc");
        string source11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "diagnostics.inc");
        string errorRecorderBoundary = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11Diagnostics.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11FifteenthBatch.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngineInspector.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngineInspector.Trt11Diagnostics.cs");

        foreach (string manifest in new[] { manifest8, manifest10, manifest11Deployment })
        {
            Assert.Contains("engine-inspector-get-engine-information", manifest);
            Assert.Contains("engine_inspector_get_engine_information", manifest);
            Assert.Contains("\"name\":  \"output_buffer\"", manifest);
            Assert.Contains("\"type\":  \"char*\"", manifest);
            Assert.Contains("\"managedType\":  \"IntPtr\"", manifest);
            Assert.Contains("\"out_required_size\"", manifest);
        }

        Assert.Contains("engine-inspector-get-layer-information", manifest8);
        Assert.Contains("engine-inspector-get-layer-information", manifest10);
        Assert.Contains("engine-inspector-get-layer-information", manifest11Eleventh);
        Assert.Contains("engine-inspector-has-error-recorder", errorRecorderManifest8);
        Assert.Contains("engine-inspector-clear-error-recorder", errorRecorderManifest8);
        Assert.Contains("engine-inspector-has-error-recorder", errorRecorderManifest10);
        Assert.Contains("engine-inspector-clear-error-recorder", errorRecorderManifest10);
        Assert.Contains("engine-inspector-has-error-recorder", manifest11Eleventh);
        Assert.Contains("engine-inspector-clear-error-recorder", ReadSource("native", "manifests", "tensorrt", "v11", "trt11-fifteenth-batch.manifest.json"));
        Assert.Contains("engine-inspector-get-error-recorder-deferred", deferred8);
        Assert.Contains("engine-inspector-set-error-recorder-deferred", deferred8);
        Assert.Contains("engine-inspector-get-error-recorder-deferred", deferred10);
        Assert.Contains("engine-inspector-set-error-recorder-deferred", deferred10);

        Assert.Contains("copy_engine_information(inspector_payload, format, output_buffer, output_buffer_size, out_required_size)", source8);
        Assert.Contains("copy_engine_information(inspector_payload, format, output_buffer, output_buffer_size, out_required_size)", source10);
        Assert.Contains("copy_string_to_buffer(text, output_buffer, output_buffer_size, out_required_size)", source8);
        Assert.Contains("copy_string_to_buffer(text, output_buffer, output_buffer_size, out_required_size)", source10);
        Assert.Contains("copy_string_to_buffer(inspector_payload->getLayerInformation", source11);
        Assert.Contains("engine_inspector_has_error_recorder_with_seh_guard", errorRecorderBoundary);
        Assert.Contains("engine_inspector_clear_error_recorder_with_seh_guard", errorRecorderBoundary);
        Assert.Contains("JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_inspector_has_error_recorder)", errorRecorderBoundary);
        Assert.Contains("JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_inspector_clear_error_recorder)", errorRecorderBoundary);
        Assert.Contains("trt11_engine_inspector_has_error_recorder_with_seh_guard", source11);

        Assert.Contains("public string GetEngineInformation", wrapper);
        Assert.Contains("public string GetLayerInformation", wrapper);
        Assert.Contains("public bool HasErrorRecorder", wrapper);
        Assert.Contains("NativeBridgeApi.GetEngineInformation(Line, _handle, format)", wrapper);
        Assert.Contains("NativeBridgeApi.GetEngineInspectorLayerInformation(Line, _handle, layerIndex, format)", wrapper);
        Assert.DoesNotContain("public IntPtr", wrapper);
        Assert.DoesNotContain("public nint", wrapper);

        Assert.Contains("ReadUtf8Buffer", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_layer_information", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_layer_information", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_layer_information", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_has_error_recorder", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_has_error_recorder", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_has_error_recorder", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_clear_error_recorder", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_clear_error_recorder", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_clear_error_recorder", interop);
        Assert.DoesNotContain("EnsureTensorRt11DeploymentApi(line, nameof(HasEngineInspectorErrorRecorder))", interop);
    }

    [Fact]
    public void RefitWeightsSmokeRunnerReportsRefitterAndInspectorDiagnosticsAsSkippableEvidence()
    {
        string program = ReadSource("smoke", "RefitWeightsSmokeRunner", "Program.cs");
        string solution = ReadSource("TensorRtSharp.sln");
        string smokeReadme = ReadSource("smoke", "README.md");

        Assert.Contains("RefitWeightsSmokeRunner", smokeReadme);
        Assert.Contains("RefitWeightsSmokeRunner.csproj", solution);
        Assert.Contains("ProbeRefitterDiagnostics(refitter)", program);
        Assert.Contains("RefitterDiagnosticSnapshot", ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitterDiagnosticSnapshot.cs"));
        Assert.Contains("refitter.GetDiagnosticSnapshot()", ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs"));
        Assert.Contains("ProbeParserRefitterControls(refitter, logger)", ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs"));
        Assert.Contains("TensorRtOnnxParserRefitterDiagnosticSnapshot snapshot = parserRefitter.GetDiagnosticSnapshot();", ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs"));
        Assert.Contains("ParserRefitterDiagnosticSnapshot=", ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs"));
        Assert.Contains("ProbeEngineInspectorDiagnostics(inspector)", program);
        Assert.Contains("RefitDiagnostics Refitter=", program);
        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
        Assert.Contains("PrintDependencyProbe(line)", program);
        Assert.Contains("GetDynamicRangeTensorNames()", program);
        Assert.Contains("GetMissingNamedWeights()", program);
        Assert.Contains("GetAllNamedWeights()", program);
        Assert.Contains("inspector.GetEngineInformation", program);
        Assert.Contains("inspector.GetLayerInformation", program);
        Assert.Contains("Skipped=True Reason=", program);
        Assert.Contains("structured exception with code 3228369022", program);
        Assert.Contains("catch (Exception exception) when (IsSkippableEnvironmentException(exception))", program);
        Assert.Contains("DllNotFoundException", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
