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
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.ObjectCreation.cs");
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

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(segments).ToArray()));
    }
}
