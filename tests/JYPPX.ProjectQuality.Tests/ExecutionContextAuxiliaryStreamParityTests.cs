using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ExecutionContextAuxiliaryStreamParityTests
{
    [Fact]
    public void TensorRt8And10ManifestsAddSetAndClearWithoutDeletingDeferredHistory()
    {
        string manifest8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-execution-context-auxiliary-streams.manifest.json");
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-execution-context-auxiliary-streams.manifest.json");
        string deferred8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Equal(2, CountOccurrences(manifest8, "\"entryPoint\""));
        Assert.Equal(2, CountOccurrences(manifest10, "\"entryPoint\""));
        Assert.Contains("trt8-execution-context-set-aux-streams", manifest8, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-clear-aux-streams", manifest8, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8 && JYPPX_HAS_CUDA_TOOLKIT", manifest8, StringComparison.Ordinal);
        Assert.Contains("trt10-execution-context-set-aux-streams", manifest10, StringComparison.Ordinal);
        Assert.Contains("trt10-execution-context-clear-aux-streams", manifest10, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10 && JYPPX_HAS_CUDA_TOOLKIT", manifest10, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-set-aux-streams-deferred", deferred8, StringComparison.Ordinal);
        Assert.Contains("trt10-execution-context-set-aux-streams-deferred", deferred10, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedNativeImplementationValidatesStreamsAndContainsExceptionBoundaries()
    {
        string common = ReadSource("native", "src", "tensorrt", "common", "execution_context_auxiliary_streams.inc");
        string api8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string api11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string oldSet11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "fourteenth_batch_controls.inc");
        string oldClear11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "diagnostics.inc");

        Assert.Contains("kMaximumAuxiliaryStreamCount", common, StringComparison.Ordinal);
        Assert.Contains("validate_stream", common, StringComparison.Ordinal);
        Assert.Contains("stream_payload->handle == nullptr", common, StringComparison.Ordinal);
        Assert.Contains("Auxiliary streams must be unique", common, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", common, StringComparison.Ordinal);
        Assert.Contains("report_vendor_seh_exception", common, StringComparison.Ordinal);
        Assert.Contains("catch (const std::exception& exception)", common, StringComparison.Ordinal);
        Assert.Contains("catch (...)", common, StringComparison.Ordinal);
        Assert.Contains("report_vendor_exception", common, StringComparison.Ordinal);

        foreach (string api in new[] { api8, api10, api11 })
        {
            Assert.Contains("execution_context_auxiliary_streams.inc", api, StringComparison.Ordinal);
            Assert.Contains("JYPPX_TRT_EXECUTION_CONTEXT_SET_AUX_STREAMS_API", api, StringComparison.Ordinal);
            Assert.Contains("JYPPX_TRT_EXECUTION_CONTEXT_CLEAR_AUX_STREAMS_API", api, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("jyppx_trt11_execution_context_set_aux_streams(", oldSet11, StringComparison.Ordinal);
        Assert.DoesNotContain("jyppx_trt11_execution_context_clear_aux_streams(", oldClear11, StringComparison.Ordinal);
    }

    [Fact]
    public void HeadersGeneratedInteropAndManagedRoutingCoverAllThreeTensorRtLines()
    {
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string generated = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string setInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11FourteenthBatch.cs");
        string clearInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Execution", "NativeBridgeApi.ExecutionContextDiagnostics.cs");

        foreach ((string header, string line) in new[] { (header8, "8"), (header10, "10"), (header11, "11") })
        {
            Assert.Contains($"jyppx_trt{line}_execution_context_set_aux_streams", header, StringComparison.Ordinal);
            Assert.Contains($"jyppx_trt{line}_execution_context_clear_aux_streams", header, StringComparison.Ordinal);
            Assert.Contains($"jyppx_trt{line}_execution_context_set_aux_streams", generated, StringComparison.Ordinal);
            Assert.Contains($"jyppx_trt{line}_execution_context_clear_aux_streams", generated, StringComparison.Ordinal);
        }

        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_aux_streams", setInterop, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_aux_streams", setInterop, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_aux_streams", setInterop, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_aux_streams", clearInterop, StringComparison.Ordinal);
        Assert.DoesNotContain("EnsureTensorRt11DeploymentApi(line, nameof(SetExecutionContextAuxStreams))", setInterop, StringComparison.Ordinal);
        Assert.DoesNotContain("EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextAuxStreams))", clearInterop, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSafeHandleLeaseSpansAssignmentUntilClearOrContextTeardown()
    {
        string lease = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Handles", "TensorRtAuxiliaryStreamHandleLease.cs");
        string controls = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11DeploymentControls.cs");
        string clear = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string context = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");

        Assert.Contains("handle.DangerousAddRef(ref addedRef)", lease, StringComparison.Ordinal);
        Assert.Contains("handles[i].DangerousRelease()", lease, StringComparison.Ordinal);
        Assert.Contains("handle.IsClosed || handle.IsInvalid", lease, StringComparison.Ordinal);
        Assert.Contains("HashSet<IntPtr>", lease, StringComparison.Ordinal);
        Assert.Contains("Auxiliary CUDA streams must be unique", lease, StringComparison.Ordinal);

        int nativeSet = controls.IndexOf("NativeBridgeApi.SetExecutionContextAuxStreams", StringComparison.Ordinal);
        int installLease = controls.IndexOf("_auxiliaryStreamLease = pendingLease", StringComparison.Ordinal);
        Assert.True(nativeSet >= 0 && installLease > nativeSet, "The persistent lease must be installed only after native set succeeds.");

        int nativeClear = clear.IndexOf("NativeBridgeApi.ClearExecutionContextAuxStreams", StringComparison.Ordinal);
        int releasePrevious = clear.IndexOf("previousLease?.Dispose()", StringComparison.Ordinal);
        Assert.True(nativeClear >= 0 && releasePrevious > nativeClear, "The old lease must be released only after native clear succeeds.");

        int contextTeardown = context.IndexOf("_handle.Dispose()", StringComparison.Ordinal);
        int releaseOnDispose = context.IndexOf("auxiliaryStreamLease?.Dispose()", StringComparison.Ordinal);
        Assert.True(contextTeardown >= 0 && releaseOnDispose > contextTeardown, "The native context must be torn down before auxiliary stream leases are released.");
        Assert.DoesNotContain("stream.Dispose()", controls + clear + context, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicSnapshotAndWrappersRemainPointerFree()
    {
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtAuxiliaryStreamAssignmentSnapshot.cs");
        string controls = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11DeploymentControls.cs");
        string clear = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");

        Assert.Contains("public void SetAuxStreams", controls, StringComparison.Ordinal);
        Assert.Contains("public TensorRtAuxiliaryStreamAssignmentSnapshot GetAuxiliaryStreamAssignmentSnapshot()", controls, StringComparison.Ordinal);
        Assert.Contains("public void ClearAuxStreams()", clear, StringComparison.Ordinal);
        Assert.Contains("public bool NativeStreamPointerExposed => false", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool BorrowedHandleEscaped => false", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool ManagedHandleLeaseActive", snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", snapshot + controls + clear, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", snapshot + controls + clear, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", snapshot + controls + clear, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeCudaStreamHandle", snapshot + controls + clear, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeConsumerReadinessAndDocumentationExposeLifetimeEvidence()
    {
        string smoke = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string docs = ReadSource("docs", "articles", "zh-cn", "execution-context-auxiliary-stream-lifetime.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("context.ClearAuxStreams()", smoke, StringComparison.Ordinal);
        Assert.Contains("AuxiliaryStreams=Line:", smoke, StringComparison.Ordinal);
        Assert.Contains("GetAuxiliaryStreamAssignmentSnapshot", smoke, StringComparison.Ordinal);
        Assert.Contains("execution-context-auxiliary-stream-lifetime", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtExecutionContext.SetAuxStreams)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.BorrowedHandleEscaped)", consumer, StringComparison.Ordinal);
        Assert.Contains("name = \"execution-context-auxiliary-stream-lifetime\"", readiness, StringComparison.Ordinal);
        Assert.Contains("hasExecutionContextAuxiliaryStreamLifetime", readiness, StringComparison.Ordinal);
        Assert.Contains("managed-safehandle-lease", readiness, StringComparison.Ordinal);
        Assert.Contains("DangerousAddRef", docs, StringComparison.Ordinal);
        Assert.Contains("DangerousRelease", docs, StringComparison.Ordinal);
        Assert.Contains("execution-context-auxiliary-stream-lifetime.md", toc, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageAliasPromotesTensorRt8And10WhileRetainingDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");

        Assert.Contains("\"IExecutionContext::setAuxStreams\" = @(\"id:*execution-context-set-aux-streams\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::setAuxStreams\" = @(\"id:*execution-context-set-aux-streams-deferred\")", script, StringComparison.Ordinal);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int explicitPriorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", explicitPriorityStart, StringComparison.Ordinal);
        string explicitPriorityBlock = script.Substring(explicitPriorityStart, heuristicStart - explicitPriorityStart);
        Assert.Contains("\"IExecutionContext::setAuxStreams\"", explicitPriorityBlock, StringComparison.Ordinal);

        Assert.Contains("\"IExecutionContext\",\"setAuxStreams\",\"IExecutionContext::setAuxStreams\",\"engine-context\",\"implemented-with-deferred-history\"", coverage, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-set-aux-streams;trt8-execution-context-set-aux-streams-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("trt10-execution-context-set-aux-streams;trt10-execution-context-set-aux-streams-deferred", coverage, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string value, string marker)
    {
        int count = 0;
        int index = 0;
        while ((index = value.IndexOf(marker, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += marker.Length;
        }

        return count;
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }
}
