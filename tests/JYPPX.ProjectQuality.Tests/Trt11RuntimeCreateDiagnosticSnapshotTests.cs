using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11RuntimeCreateDiagnosticSnapshotTests
{
    [Fact]
    public void RuntimeCreateDiagnosticUsesPointerFreeSnapshotAndNoThrowNativeEntry()
    {
        string nativeTypes = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string trt11Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string trt11Source = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string nativeStructs = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeStructs.cs");
        string nativeMethods = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeMethodsTensorRt.cs");
        string bridgeApi = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Runtime", "NativeBridgeApi.RuntimeCreation.cs");
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntimeCreateDiagnosticSnapshot.cs");
        string runtimeConsumer = ReadSource("eng", "Test-BridgePackageRuntimeConsumer.ps1");

        Assert.Contains("JYPPX_TensorRtRuntimeCreateDiagnosticInfo", nativeTypes);
        Assert.Contains("logger_callback_available", nativeTypes);
        Assert.Contains("logger_message_count", nativeTypes);
        Assert.Contains("last_logger_message", nativeTypes);
        Assert.Contains("create_runtime_phase", nativeTypes);
        Assert.Contains("native_detail", nativeTypes);
        Assert.Contains("jyppx_trt11_runtime_create_diagnostic", trt11Header);
        Assert.Contains("reset_runtime_create_diagnostic_info", trt11Source);
        Assert.Contains("create_infer_runtime_with_guard(*logger_payload, &runtime)", trt11Source);
        Assert.Contains("copy_logger_runtime_create_diagnostic", trt11Source);
        Assert.Contains("after-createInferRuntime-null-guard-ok", trt11Source);
        Assert.Contains("delete runtime;", trt11Source);
        Assert.Contains("return JYPPX_STATUS_OK;", trt11Source);
        Assert.Contains("NativeTensorRtRuntimeCreateDiagnosticInfo", nativeStructs);
        Assert.Contains("jyppx_trt11_runtime_create_diagnostic", nativeMethods);
        Assert.Contains("GetRuntimeCreateDiagnostic", bridgeApi);
        Assert.Contains("public static TensorRtRuntimeCreateDiagnosticSnapshot GetRuntimeCreateDiagnostic", environmentProbe);

        Assert.Contains("public sealed class TensorRtRuntimeCreateDiagnosticSnapshot", snapshot);
        Assert.Contains("public bool DiagnosticAvailable", snapshot);
        Assert.Contains("public bool Attempted", snapshot);
        Assert.Contains("public bool CreateInferRuntimeReturnedNull", snapshot);
        Assert.Contains("public BridgeStatusCode LastStatus", snapshot);
        Assert.Contains("public bool LoggerCallbackAvailable", snapshot);
        Assert.Contains("public uint LoggerMessageCount", snapshot);
        Assert.Contains("public string LastLoggerMessage", snapshot);
        Assert.Contains("public string CreateRuntimePhase", snapshot);
        Assert.Contains("public string NativeDetail", snapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof => false", snapshot);
        Assert.Contains("public bool IsPackageConsumerRuntimeProof => false", snapshot);
        Assert.Contains("public bool ExposesNativePointer => false", snapshot);
        Assert.DoesNotContain("public IntPtr", snapshot);
        Assert.DoesNotContain("public nint", snapshot);
        Assert.DoesNotContain("public UIntPtr", snapshot);

        Assert.Contains("WriteRuntimeCreateDiagnostic(line)", runtimeConsumer);
        Assert.Contains("typeof(TensorRtEnvironmentProbe).GetMethod", runtimeConsumer);
        Assert.Contains("Runtime create diagnostic API is not available in this managed package.", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeDiagnosticAvailable=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeAttempted=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeReturnedNull=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeLastStatus=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeLoggerCallbackAvailable=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeLoggerMessageCount=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeLastLoggerMessage=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimePhase=", runtimeConsumer);
        Assert.Contains("NativeCreateRuntimeNativeDetail=", runtimeConsumer);
        Assert.Contains("runtimeCreateDiagnostic = $runtimeCreateDiagnostic", runtimeConsumer);
    }

    [Fact]
    public void FinalReportsConsumeRuntimeCreateDiagnosticWithoutPromotingProof()
    {
        string rootCause = ReadSource("eng", "Export-Trt11RuntimeSmokeRootCauseReport.ps1");
        string dllResolution = ReadSource("eng", "Export-Trt11RuntimeDllResolutionReport.ps1");
        string diff = ReadSource("eng", "Export-Trt10VsTrt11BridgeRuntimeDiagnosticDiff.ps1");
        string dashboard = ReadSource("eng", "Export-FinalProofReadinessBlockerDashboard.ps1");
        string releaseBundle = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");

        foreach (string source in new[] { rootCause, dllResolution, diff, dashboard, releaseBundle })
        {
            Assert.True(
                source.Contains("nativeCreateRuntime", StringComparison.OrdinalIgnoreCase) ||
                source.Contains("runtimeCreateDiagnostic", StringComparison.OrdinalIgnoreCase),
                "Each final proof/report script must consume the TRT11 runtime-create diagnostic fields.");
        }

        Assert.Contains("runtimeCreateDiagnostic = $runtimeCreateDiagnostic", rootCause);
        Assert.Contains("nativeCreateRuntimeDiagnosticAvailable", rootCause);
        Assert.Contains("nativeCreateRuntimePhase", rootCause);
        Assert.Contains("nativeCreateRuntimeLoggerMessageCount", rootCause);
        Assert.Contains("nativeCreateRuntimeLastLoggerMessage", rootCause);
        Assert.Contains("native create-runtime diagnostic available", rootCause);
        Assert.Contains("runtimeCreateDiagnostic = $runtimeCreateDiagnostic", dllResolution);
        Assert.Contains("nativeCreateRuntimeDiagnosticAvailable", dllResolution);
        Assert.Contains("nativeCreateRuntimePhase", dllResolution);
        Assert.Contains("nativeCreateRuntimeLoggerMessageCount", dllResolution);
        Assert.Contains("runtimeCreateDiagnostic.available", diff);
        Assert.Contains("runtimeCreateDiagnostic.phase", diff);
        Assert.Contains("runtimeCreateDiagnostic.lastLoggerMessage", diff);
        Assert.Contains("trt11RuntimeCreateDiagnostic", diff);
        Assert.Contains("TRT11 native create-runtime diagnostic snapshot", dashboard);
        Assert.Contains("trt11RootCauseNativeCreateRuntimeDiagnosticAvailable", releaseBundle);
        Assert.Contains("trt11RootCauseNativeCreateRuntimePhase", releaseBundle);
        Assert.Contains("trt11RootCauseNativeCreateRuntimeLoggerMessageCount", releaseBundle);
        Assert.Contains("nativeCreateDiag=", releaseBundle);
    }

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(segments).ToArray()));
    }
}
