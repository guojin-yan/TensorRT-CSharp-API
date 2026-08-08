using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ExecutionContextErrorBufferCopyTests
{
    [Fact]
    public void ManifestNativeAndGeneratedInteropRetainOnlyADeferredCompatibilityBoundary()
    {
        string manifest = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-execution-context-error-buffer-copy.manifest.json");
        string deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-twenty-third-batch-deferred-coverage.manifest.json");
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string native8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string generated = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string bridge = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Execution", "NativeBridgeApi.ExecutionContextDeploymentMetadata.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");

        Assert.Contains("trt8-execution-context-get-error-buffer-copy-deferred", manifest, StringComparison.Ordinal);
        Assert.Contains("caller-owned", manifest, StringComparison.Ordinal);
        Assert.Contains("trt8-execution-context-get-error-buffer-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt8_execution_context_get_error_buffer_copy", header8, StringComparison.Ordinal);
        Assert.DoesNotContain("jyppx_trt10_execution_context_get_error_buffer_copy", header10, StringComparison.Ordinal);
        Assert.DoesNotContain("jyppx_trt11_execution_context_get_error_buffer_copy", header11, StringComparison.Ordinal);
        Assert.DoesNotContain("context_payload->getErrorBuffer()", native8, StringComparison.Ordinal);
        Assert.Contains("*out_required_size = 0", native8, StringComparison.Ordinal);
        Assert.Contains("report_not_implemented", native8, StringComparison.Ordinal);
        Assert.Contains("unavailable on the standard nvinfer1::IExecutionContext vendor type", native8, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt8_execution_context_get_error_buffer_copy", generated, StringComparison.Ordinal);
        Assert.Contains("GetExecutionContextErrorBuffer", bridge, StringComparison.Ordinal);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt8_execution_context_get_error_buffer_copy", bridge, StringComparison.Ordinal);
        Assert.Contains("TryGetErrorBuffer", wrapper, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", wrapper, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", wrapper, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", wrapper, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeReportsDeferredDiagnosticWithoutClaimingRuntimeProof()
    {
        string smoke = ReadSource("smoke", "NetworkActivationPoolingResizeSmokeRunner", "Program.cs");
        Assert.Contains("TryGetErrorBuffer", smoke, StringComparison.Ordinal);
        Assert.Contains("ExecutionContextErrorBufferDeferredDiagnostic Available=", smoke, StringComparison.Ordinal);
        Assert.DoesNotContain("RuntimeProof=True", smoke, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }
}
