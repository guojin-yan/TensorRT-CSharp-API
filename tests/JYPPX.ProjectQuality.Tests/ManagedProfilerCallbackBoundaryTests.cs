using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedProfilerCallbackBoundaryTests
{
    [Fact]
    public void ManagedProfilerCallbackApisAreManifestedForAllTensorRtLines()
    {
        string trt8 = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-minimal.manifest.json");
        string trt10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-minimal.manifest.json");
        string trt11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-minimal.manifest.json");

        Assert.Contains("trt8-profiler-create-with-callback", trt8);
        Assert.Contains("trt8-profiler-emit-diagnostic", trt8);
        Assert.Contains("trt8-execution-context-set-profiler", trt8);
        Assert.Contains("trt8-execution-context-clear-profiler", trt8);
        Assert.Contains("trt8-execution-context-has-profiler", trt8);

        Assert.Contains("trt10-profiler-create-with-callback", trt10);
        Assert.Contains("trt10-profiler-emit-diagnostic", trt10);
        Assert.Contains("trt10-execution-context-set-profiler", trt10);
        Assert.Contains("trt10-execution-context-clear-profiler", trt10);
        Assert.Contains("trt10-execution-context-has-profiler", trt10);

        Assert.Contains("trt11-profiler-create-with-callback", trt11);
        Assert.Contains("trt11-profiler-emit-diagnostic", trt11);
        Assert.Contains("trt11-execution-context-set-profiler", trt11);

        Assert.Contains("JYPPX_TensorRtProfilerCallback", trt8);
        Assert.Contains("JYPPX_TensorRtProfilerCallback", trt10);
        Assert.Contains("JYPPX_TensorRtProfilerCallback", trt11);
    }

    [Fact]
    public void ManagedProfilerInterfaceInfoIsReadOnlyAndTensorRt11Only()
    {
        string trt10InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-callback-interface-info.manifest.json");
        string trt11InterfaceManifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-callback-interface-info.manifest.json");
        string nativeSource = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string interopSource = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string profilerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfiler.cs");

        Assert.DoesNotContain("trt10-profiler-get-interface-info", trt10InterfaceManifest);
        Assert.Contains("trt11-profiler-get-interface-info", trt11InterfaceManifest);
        Assert.Contains("\"managedType\": \"byte[]\"", trt11InterfaceManifest);
        Assert.Contains("out_required_size", trt11InterfaceManifest);
        Assert.Contains("out_major", trt11InterfaceManifest);
        Assert.Contains("out_minor", trt11InterfaceManifest);

        Assert.Contains("jyppx_trt11_profiler_get_interface_info", nativeSource);
        Assert.Contains("copy_interface_info_to_buffer", nativeSource);
        Assert.Contains("get_callback_interface_info(profiler_payload", nativeSource);

        Assert.Contains("GetProfilerInterfaceInfo", interopSource);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt11_profiler_get_interface_info", interopSource);
        Assert.Contains("throw UnsupportedCallbackInterfaceInfoLine", interopSource);

        Assert.Contains("public TensorRtInterfaceInfo InterfaceInfo", profilerSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)", profilerSource);
        Assert.Contains("public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", profilerSource);
        Assert.Contains("TensorRtProfilerInterfaceMetadataSnapshot", profilerSource);
        Assert.Contains("GetInterfaceMetadataSnapshot", profilerSource);
        Assert.Contains("TensorRT 8 and TensorRT 10 profilers are not versioned interfaces", profilerSource);
        Assert.DoesNotContain("public IntPtr", profilerSource);
        Assert.DoesNotContain("public nint", profilerSource);

        string snapshotSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfilerInterfaceMetadataSnapshot.cs");
        Assert.Contains("public readonly struct TensorRtProfilerInterfaceMetadataSnapshot", snapshotSource);
        Assert.Contains("public bool IsRuntimeProof => false", snapshotSource);
        Assert.DoesNotContain("public IntPtr", snapshotSource);
        Assert.DoesNotContain("public nint", snapshotSource);
    }

    [Fact]
    public void ManagedProfilerPublicApiOwnsCallbackStateWithoutExposingNativePointers()
    {
        string profilerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfiler.cs");
        string contextSource = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");
        string profileSource = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Profile.cs");
        string diagnosticsSource = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");

        Assert.Contains("public delegate void TensorRtProfilerHandler", profilerSource);
        Assert.Contains("public sealed class TensorRtProfiler", profilerSource);
        Assert.Contains("GCHandle.Alloc(_callbackState)", profilerSource);
        Assert.Contains("~TensorRtProfiler()", profilerSource);
        Assert.Contains("AttachBorrower", profilerSource);
        Assert.Contains("DetachBorrower", profilerSource);
        Assert.Contains("LastCallbackException", profilerSource);
        Assert.Contains("they never cross the native ABI boundary", profilerSource);
        Assert.DoesNotContain("public IntPtr", profilerSource);
        Assert.DoesNotContain("public nint", profilerSource);

        Assert.Contains("_profilerKeepAlive", contextSource);
        Assert.Contains("DetachProfiler();", contextSource);
        Assert.Contains("public void SetProfiler(TensorRtProfiler profiler)", profileSource);
        Assert.Contains("NativeBridgeApi.SetExecutionContextProfiler", profileSource);
        Assert.Contains("DetachProfiler();", diagnosticsSource);
        Assert.Contains("public bool HasProfiler => _profilerKeepAlive != null", diagnosticsSource);
        Assert.Contains("public bool HasNativeProfiler => NativeBridgeApi.HasExecutionContextProfiler", diagnosticsSource);
        Assert.Contains("must not be used as a managed ownership signal", diagnosticsSource);
    }

    [Fact]
    public void ExecutionContextDisposeKeepsBorrowedProfilerAliveThroughNativeContextRelease()
    {
        string contextSource = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.cs");

        int clearIndex = contextSource.IndexOf("TryClearProfilerForDispose();", StringComparison.Ordinal);
        int disposeIndex = contextSource.IndexOf("_handle.Dispose();", StringComparison.Ordinal);
        int keepAliveIndex = contextSource.IndexOf("GC.KeepAlive(profiler);", StringComparison.Ordinal);
        int detachIndex = contextSource.IndexOf("DetachProfiler();", StringComparison.Ordinal);

        Assert.True(clearIndex >= 0, "Dispose should first try to clear the native borrowed profiler pointer.");
        Assert.True(disposeIndex > clearIndex, "The native context handle must be released after the native clear attempt.");
        Assert.True(keepAliveIndex > disposeIndex, "The profiler must be kept alive through native context release.");
        Assert.True(detachIndex > keepAliveIndex, "The managed borrow count must be detached only after the context handle is released.");
        Assert.Contains("catch (BridgeProbeException)", contextSource);
        Assert.Contains("TensorRT never observes a freed borrowed profiler", contextSource);
    }

    [Fact]
    public void ManagedProfilerBridgeApisAreDeclaredInPublicNativeHeaders()
    {
        AssertProfilerHeaderDeclarations("8");
        AssertProfilerHeaderDeclarations("10");
        AssertProfilerHeaderDeclarations("11");
    }

    [Fact]
    public void NativeManagedProfilerTrampolineSwallowsCallbackFailures()
    {
        string nativeSources = string.Join(
            Environment.NewLine,
            ReadSource("native", "src", "tensorrt", "v8", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v10", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v11", "api.cpp"));

        Assert.Contains("class ManagedProfiler final", nativeSources);
        Assert.Contains("JYPPX_TensorRtProfilerCallback callback_", nativeSources);
        Assert.Contains("callback_status != JYPPX_STATUS_OK", nativeSources);
        Assert.Contains("last_callback_failed_ = true", nativeSources);
        Assert.Contains("Managed TensorRT profiler callback returned status", nativeSources);
        Assert.Contains("catch (const std::exception& exception)", nativeSources);
        Assert.Contains("catch (...)", nativeSources);
    }

    [Fact]
    public void ManagedProfilerCallbackSmokeProvidesSkippableConsumerCoverage()
    {
        string project = ReadSource("smoke", "ManagedProfilerCallbackSmokeRunner", "ManagedProfilerCallbackSmokeRunner.csproj");
        string program = ReadSource("smoke", "ManagedProfilerCallbackSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("ManagedProfilerCallbackSmokeRunner", smokeReadme);
        Assert.Contains("ManagedProfilerCallbackSmokeRunner.csproj", solution);
        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj\" />", project);

        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
        Assert.Contains("new TensorRtProfiler(", program);
        Assert.Contains("EmitDiagnostic", program);
        Assert.Contains("profiler.TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)", program);
        Assert.Contains("ManagedProfilerInterfaceInfo Skipped=True Reason={diagnostic}", program);
        Assert.Contains("ManagedProfilerInterfaceInfo", program);
        Assert.Contains("GetInterfaceMetadataSnapshot", program);
        Assert.Contains("ManagedProfilerInterfaceMetadataSnapshot", program);
        Assert.Contains("RuntimeProof={metadataSnapshot.IsRuntimeProof}", program);
        Assert.Contains("ManagedProfilerCallbackException", program);
        Assert.Contains("SetProfiler", program);
        Assert.Contains("ClearProfiler", program);
        Assert.Contains("HasNativeProfiler", program);
        Assert.Contains("Native=", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertProfilerHeaderDeclarations(string line)
    {
        string header = ReadSource("native", "include", "jyppx", "tensorrt", $"trt{line}.h");

        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_profiler_create_with_callback", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_profiler_emit_diagnostic", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_execution_context_set_profiler", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_execution_context_clear_profiler", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_execution_context_has_profiler", header);
    }
}
