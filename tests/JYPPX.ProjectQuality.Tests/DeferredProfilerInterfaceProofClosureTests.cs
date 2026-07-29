using System;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredProfilerInterfaceProofClosureTests
{
    [Fact]
    public void ProofClosureArtifactKeepsRuntimeAndDeferredBoundariesExplicit()
    {
        string json = ReadSource("artifacts", "interface-coverage", "trt11-profiler-interface-info-proof-closure.json");
        string markdown = ReadSource("artifacts", "interface-coverage", "trt11-profiler-interface-info-proof-closure.md");

        Assert.Contains("IProfiler::getInterfaceInfo", json, StringComparison.Ordinal);
        Assert.Contains("trt11-profiler-get-interface-info", json, StringComparison.Ordinal);
        Assert.Contains("trt11-profiler-get-interface-info-deferred", json, StringComparison.Ordinal);
        Assert.Contains("\"isRuntimeExecutionProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"canDeleteDeferredRecord\": false", json, StringComparison.Ordinal);
        Assert.Contains("controlled unsupported", markdown, StringComparison.Ordinal);
        Assert.Contains("caller buffer + scalar out", markdown, StringComparison.Ordinal);
        Assert.Contains("不证明", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeManagedAndDeferredSourcesFormTheProfilerVersionMatrix()
    {
        string manifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-callback-interface-info.manifest.json");
        string native = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string deferred = ReadSource("native", "src", "tensorrt", "v11", "modules", "deferred", "twenty_third_batch_deferred.inc");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Callbacks", "NativeBridgeApi.CallbackInterfaceInfo.cs");
        string profiler = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfiler.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtProfilerInterfaceMetadataSnapshot.cs");

        Assert.Contains("trt11-profiler-get-interface-info", manifest, StringComparison.Ordinal);
        Assert.Contains("trt11-profiler-get-api-language", manifest, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_profiler_get_interface_info", native, StringComparison.Ordinal);
        Assert.Contains("copy_interface_info_to_buffer", native, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_profiler_get_interface_info_deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("TensorRT 11 only", interop, StringComparison.Ordinal);
        Assert.Contains("GetInterfaceMetadataSnapshot", profiler, StringComparison.Ordinal);
        Assert.Contains("public bool IsRuntimeProof => false", snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", snapshot, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        RepositorySourceReader.Read(System.IO.Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
