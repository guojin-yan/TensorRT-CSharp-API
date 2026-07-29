using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ExecutionContextReadonlyControlsTests
{
    [Fact]
    public void ManagedExecutionContextRuntimeDiagnosticSnapshotAggregatesSafeReadonlySignals()
    {
        string runtimeDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string deploymentBuilder = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11DeploymentSnapshot.cs");
        string deploymentSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContextDeploymentSnapshot.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContextRuntimeDiagnosticSnapshot.cs");
        string callbackSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContextCallbackStateSnapshot.cs");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string tensorRtSmoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");

        Assert.Contains("public TensorRtExecutionContextRuntimeDiagnosticSnapshot GetRuntimeDiagnosticSnapshot(string outputTensorName)", runtimeDiagnostics);
        Assert.Contains("GetCallbackStateSnapshot(outputTensorName)", runtimeDiagnostics);
        Assert.Contains("HasErrorRecorder", runtimeDiagnostics);
        Assert.Contains("IsInputConsumedEventSet", runtimeDiagnostics);
        Assert.Contains("InputConsumedEventAddressValue", runtimeDiagnostics);
        Assert.Contains("HasOutputAllocator(outputTensorName)", runtimeDiagnostics);
        Assert.Contains("IsOutputTensorAddressSet(outputTensorName)", runtimeDiagnostics);
        Assert.Contains("GetOutputTensorAddressValue(outputTensorName)", runtimeDiagnostics);
        Assert.Contains("HasTemporaryStorageAllocator", runtimeDiagnostics);
        Assert.Contains("HasDebugListener", runtimeDiagnostics);
        Assert.Contains("HasNativeProfiler", runtimeDiagnostics);
        Assert.Contains("HasRuntimeConfig", runtimeDiagnostics);
        Assert.Contains("GetNvtxVerbosity", runtimeDiagnostics);
        Assert.Contains("GetUnfusedTensorsDebugState", runtimeDiagnostics);
        Assert.Contains("CreateUnavailableCallbackStateSnapshot", runtimeDiagnostics);

        Assert.Contains("public sealed class TensorRtExecutionContextRuntimeDiagnosticSnapshot", snapshot);
        Assert.Contains("public TensorRtExecutionContextCallbackStateSnapshot CallbackState", snapshot);
        Assert.Contains("public IReadOnlyList<string> Diagnostics", snapshot);
        Assert.Contains("public TensorRtExecutionContextRuntimeDiagnosticSummary ToSummary()", snapshot);
        Assert.Contains("public sealed class TensorRtExecutionContextRuntimeDiagnosticSummary", snapshot);
        Assert.Contains("This method only reads managed snapshot values. It does not call TensorRT, expose native pointers, invoke callbacks, or promote runtime proof.", snapshot);
        Assert.Contains("public bool HasOutputTensorName { get; }", snapshot);
        Assert.Contains("public bool HasOutputAllocator { get; }", snapshot);
        Assert.Contains("public bool IsOutputTensorAddressSet { get; }", snapshot);
        Assert.Contains("public BridgeStatusCode CallbackStateLastStatus { get; }", snapshot);
        Assert.Contains("public int DiagnosticCount { get; }", snapshot);
        Assert.Contains("List<TensorRtExecutionContextRuntimeDiagnosticSnapshot> runtimeDiagnostics", deploymentBuilder);
        Assert.Contains("GetRuntimeDiagnosticSnapshot(tensor.Name)", deploymentBuilder);
        Assert.Contains("CreateUnavailableRuntimeDiagnosticSnapshot", deploymentBuilder);
        Assert.Contains("public IReadOnlyList<TensorRtExecutionContextRuntimeDiagnosticSnapshot> RuntimeDiagnostics", deploymentSnapshot);
        Assert.Contains("public TensorRtExecutionContextDeploymentSummary ToSummary()", deploymentSnapshot);
        Assert.Contains("public sealed class TensorRtExecutionContextDeploymentSummary", deploymentSnapshot);
        Assert.Contains("public int CopiedTensorStateCount", deploymentSnapshot);
        Assert.Contains("public int CopiedRuntimeDiagnosticCount", deploymentSnapshot);
        Assert.Contains("public int RuntimeDiagnosticsWithCallbackStateCount", deploymentSnapshot);
        Assert.Contains("public int RuntimeDiagnosticsWithOutputAllocatorCount", deploymentSnapshot);
        Assert.Contains("public bool CopiedTensorStatesMatchEngineIOTensorCount", deploymentSnapshot);
        Assert.Contains("public bool PointerFreeCopiedSummary", deploymentSnapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof", deploymentSnapshot);
        Assert.Contains("public bool CanDeleteDeferredRecord", deploymentSnapshot);
        Assert.Contains("runtimeDiagnostics={RuntimeDiagnostics.Count}", deploymentSnapshot);
        Assert.Contains("address values are emitted only as integer diagnostics", runtimeDiagnostics);
        Assert.Contains("不暴露任何 borrowed pointer", snapshot);
        Assert.DoesNotContain("public IntPtr", snapshot + runtimeDiagnostics + deploymentBuilder + deploymentSnapshot + callbackSnapshot);
        Assert.DoesNotContain("public nint", snapshot + runtimeDiagnostics + deploymentBuilder + deploymentSnapshot + callbackSnapshot);

        Assert.Contains("context.GetRuntimeDiagnosticSnapshot(outputTensorName)", smoke);
        Assert.Contains("ExecutionContextRuntimeDiagnosticSnapshot=GetRuntimeDiagnosticSnapshot;TensorRtExecutionContextRuntimeDiagnosticSnapshot;pointer-free", smoke);
        Assert.Contains("ExecutionContextRuntimeDiagnosticSummary=ToSummary;TensorRtExecutionContextRuntimeDiagnosticSummary;pointer-free;not-runtime-proof", smoke);
        Assert.Contains("ExecutionContextRuntimeDiagnosticSummary={runtimeSummary.HasErrorRecorder}/{runtimeSummary.HasOutputAllocator}/{runtimeSummary.IsOutputTensorAddressSet}/{runtimeSummary.HasTemporaryStorageAllocator}/{runtimeSummary.HasDebugListener}/{runtimeSummary.HasNativeProfiler}/{runtimeSummary.CallbackStateLastStatus}/{runtimeSummary.DiagnosticCount}", smoke);
        Assert.Contains("RuntimeDiagnosticSnapshot=", smoke);
        Assert.Contains("contextSnapshot.RuntimeDiagnostics.Count", tensorRtSmoke);
        Assert.Contains("ExecutionContextDeploymentSummary=", tensorRtSmoke);
    }

    [Fact]
    public void TensorRt8And10NvtxVerbosityManifestsPromoteRealAbiWithoutDeletingDeferredRecords()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-execution-context-readonly-controls.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-execution-context-readonly-controls.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("jyppx-trt8-execution-context-set-nvtx-verbosity", manifest8);
        Assert.Contains("jyppx-trt8-execution-context-get-nvtx-verbosity", manifest8);
        Assert.Contains("jyppx-trt8-engine-get-hardware-compatibility-level", manifest8);
        Assert.Contains("\"type\": \"JYPPX_TensorRtExecutionContext*\", \"direction\": \"in\"", manifest8);
        Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\"", manifest8);
        Assert.Contains("\"type\": \"int32_t*\", \"direction\": \"out\"", manifest8);
        Assert.DoesNotContain("_deferred", manifest8);

        Assert.Contains("jyppx-trt10-execution-context-set-nvtx-verbosity", manifest10);
        Assert.Contains("jyppx-trt10-execution-context-get-nvtx-verbosity", manifest10);
        Assert.DoesNotContain("_deferred", manifest10);

        Assert.Contains("trt8-execution-context-get-nvtx-verbosity-deferred", deferred8);
        Assert.Contains("trt8-execution-context-set-nvtx-verbosity-deferred", deferred8);
        Assert.Contains("trt8-cuda-engine-get-hardware-compatibility-level-deferred", deferred8);
        Assert.Contains("trt10-execution-context-get-nvtx-verbosity-deferred", deferred10);
        Assert.Contains("trt10-execution-context-set-nvtx-verbosity-deferred", deferred10);
    }

    [Fact]
    public void NativeHeadersAndSourcesExposeSafeExecutionContextNvtxAndEngineHardwareParameters()
    {
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string api8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");

        Assert.Contains("jyppx_trt8_execution_context_set_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t verbosity, JYPPX_Boolean* out_set)", header8);
        Assert.Contains("jyppx_trt8_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity)", header8);
        Assert.Contains("jyppx_trt8_engine_get_hardware_compatibility_level(JYPPX_TensorRtCudaEngine* engine, int32_t* out_level)", header8);
        Assert.Contains("jyppx_trt10_execution_context_set_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t verbosity, JYPPX_Boolean* out_set)", header10);
        Assert.Contains("jyppx_trt10_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity)", header10);

        foreach (string api in new[] { api8, api10 })
        {
            Assert.Contains("context_payload->setNvtxVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity))", api);
            Assert.Contains("context_payload->getNvtxVerbosity()", api);
            Assert.Contains("validate_output_pointer(out_set, \"out_set\")", api);
            Assert.Contains("validate_output_pointer(out_verbosity, \"out_verbosity\")", api);
        }

        Assert.Contains("engine_payload->getHardwareCompatibilityLevel()", api8);
        Assert.DoesNotContain("JYPPX_TensorRtPluginCreator**", header8 + header10);
    }

    [Fact]
    public void ManagedInteropAndPublicApiRouteNvtxAndEngineHardwareAcrossTensorRt8_10_11()
    {
        string diagnosticsInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11Diagnostics.cs");
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Engine", "NativeBridgeApi.EngineRuntimeControls.cs");
        string contextApi = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string engineApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11RuntimeControls.cs");
        string smoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_nvtx_verbosity", diagnosticsInterop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_nvtx_verbosity", diagnosticsInterop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_nvtx_verbosity", diagnosticsInterop);
        Assert.DoesNotContain("EnsureTensorRt11DeploymentApi(line, nameof(SetExecutionContextNvtxVerbosity))", diagnosticsInterop);
        Assert.DoesNotContain("EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextNvtxVerbosity))", diagnosticsInterop);

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_hardware_compatibility_level", runtimeInterop);
        Assert.Contains("TensorRT 8, 10, and 11", runtimeInterop);

        Assert.Contains("public bool SetNvtxVerbosity", contextApi);
        Assert.Contains("public TensorRtProfilingVerbosity GetNvtxVerbosity", contextApi);
        Assert.Contains("TensorRT execution context", contextApi);
        Assert.Contains("public TensorRtHardwareCompatibilityLevel EngineHardwareCompatibilityLevel", engineApi);
        Assert.Contains("TensorRT 8, TensorRT 10, and TensorRT 11", engineApi);
        Assert.DoesNotContain("public IntPtr", contextApi + engineApi);
        Assert.DoesNotContain("public nint", contextApi + engineApi);

        Assert.Contains("context.GetNvtxVerbosity()", smoke);
        Assert.Contains("context.SetNvtxVerbosity(nvtxBefore)", smoke);
        Assert.Contains("engine.EngineHardwareCompatibilityLevel", smoke);
    }

    [Fact]
    public void InterfaceCoverageMatrixSeparatesReadonlyPromotionsFromDeferredHistory()
    {
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        foreach (string matrix in new[] { coverage, comparison })
        {
            Assert.Contains("\"IExecutionContext\",\"getNvtxVerbosity\",\"IExecutionContext::getNvtxVerbosity\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"IExecutionContext\",\"setNvtxVerbosity\",\"IExecutionContext::setNvtxVerbosity\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("\"ICudaEngine\",\"getHardwareCompatibilityLevel\",\"ICudaEngine::getHardwareCompatibilityLevel\",\"engine-context\",\"implemented-with-deferred-history\"", matrix);
            Assert.Contains("jyppx_trt8_execution_context_get_nvtx_verbosity;jyppx_trt8_execution_context_get_nvtx_verbosity_deferred", matrix);
            Assert.Contains("jyppx_trt10_execution_context_set_nvtx_verbosity;jyppx_trt10_execution_context_set_nvtx_verbosity_deferred", matrix);
            Assert.Contains("jyppx_trt8_engine_get_hardware_compatibility_level", matrix);
            Assert.Contains("jyppx_trt8_cuda_engine_get_hardware_compatibility_level_deferred", matrix);
        }
    }

    [Fact]
    public void NvtxVerbosityDeferredHistoryUsesExplicitCoverageAliasesAndPointerFreeAudit()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string audit = ReadSource("artifacts", "interface-coverage", "trt-execution-context-nvtx-verbosity-candidate-audit.md");
        string auditJson = ReadSource("artifacts", "interface-coverage", "trt-execution-context-nvtx-verbosity-candidate-audit.json");

        Assert.Contains("\"IExecutionContext::getNvtxVerbosity\" = @(\"id:*execution-context-get-nvtx-verbosity-deferred\")", script);
        Assert.Contains("\"IExecutionContext::setNvtxVerbosity\" = @(\"id:*execution-context-set-nvtx-verbosity-deferred\")", script);
        Assert.Contains("TRT8", audit);
        Assert.Contains("TRT10", audit);
        Assert.Contains("TRT11", audit);
        Assert.Contains("vendor header", audit);
        Assert.Contains("import library/DLL", audit);
        Assert.Contains("retain TRT8/TRT10 deferred history", audit);
        Assert.Contains("getNvtxVerbosity", auditJson);
        Assert.Contains("setNvtxVerbosity", auditJson);
        Assert.Contains("canDeleteDeferredRecords\": false", auditJson);
        Assert.Contains("publicApiPointerFree\": true", auditJson);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
