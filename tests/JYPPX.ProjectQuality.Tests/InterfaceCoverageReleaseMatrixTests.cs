using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class InterfaceCoverageReleaseMatrixTests
{
    private static readonly string[] ClosedTensorRt11EntryPoints =
    {
        "jyppx_trt11_builder_config_get_remote_auto_tuning_config",
        "jyppx_trt11_builder_config_set_remote_auto_tuning_config",
        "jyppx_trt11_execution_context_get_unfused_tensors_debug_state",
        "jyppx_trt11_execution_context_set_unfused_tensors_debug_state",
        "jyppx_trt11_network_mark_unfused_tensors_as_debug_tensors",
        "jyppx_trt11_network_unmark_unfused_tensors_as_debug_tensors",
        "jyppx_trt11_onnx_parser_load_initializer",
        "jyppx_trt11_onnx_parser_load_model_proto",
        "jyppx_trt11_onnx_parser_parse_model_proto",
        "jyppx_trt11_parser_refitter_load_initializer",
        "jyppx_trt11_parser_refitter_load_model_proto",
        "jyppx_trt11_parser_refitter_refit_model_proto"
    };

    [Fact]
    public void CoverageBaselineSeparatesSupportedAndFutureTensorRtVersions()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-InterfaceCoverageMatrix.ps1"));
        string review = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "project-completion-review.md"));

        Assert.Contains("SupportedTensorRtVersions = @(\"8.6\", \"10.11\", \"11.0\")", script, StringComparison.Ordinal);
        Assert.Contains("ReleaseMatrixStatus", script, StringComparison.Ordinal);
        Assert.Contains("Supported TensorRT Missing By Package / Category", script, StringComparison.Ordinal);
        Assert.Contains("Future TensorRT Version Differences", script, StringComparison.Ordinal);
        Assert.Contains("10.13", review, StringComparison.Ordinal);
        Assert.Contains("future-version", review, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRt11ClosesTheTwelveLowOwnershipRiskInterfaces()
    {
        string manifests = string.Join('\n', Directory.EnumerateFiles(
                Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt", "v11"),
                "*.json",
                SearchOption.AllDirectories)
            .Select(File.ReadAllText));
        string nativeHeader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "native",
            "include",
            "jyppx",
            "tensorrt",
            "trt11.h"));
        string generatedInterop = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.Shared",
            "Generated",
            "GeneratedEntryPointNames.g.cs"));

        foreach (string entryPoint in ClosedTensorRt11EntryPoints)
        {
            Assert.Contains(entryPoint, manifests, StringComparison.Ordinal);
            Assert.Contains(entryPoint, nativeHeader, StringComparison.Ordinal);
            Assert.Contains(entryPoint, generatedInterop, StringComparison.Ordinal);
        }

        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", manifests, StringComparison.Ordinal);
    }
}
