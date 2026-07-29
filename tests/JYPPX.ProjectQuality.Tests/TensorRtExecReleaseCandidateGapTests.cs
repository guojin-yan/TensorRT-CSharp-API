using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecReleaseCandidateGapTests
{
    [Fact]
    public void TensorRtExecReleaseCandidateGapListCapturesPublishableWorkItemsAndProofBoundaries()
    {
        string jsonPath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-release-candidate-gap-list.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-release-candidate-gap-list.md");
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-release-candidate-gap-list.md");

        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));
        Assert.True(File.Exists(articlePath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;

        Assert.Equal("tensor-rt-exec-release-candidate-gap-list", root.GetProperty("gapListId").GetString());
        Assert.Equal("release-candidate-gap-planning", root.GetProperty("state").GetString());
        Assert.Contains("not package-consumer-runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);
        Assert.Equal(0, root.GetProperty("summary").GetProperty("runtimeProofItems").GetInt32());
        Assert.Equal(0, root.GetProperty("summary").GetProperty("packageConsumerRuntimeProofItems").GetInt32());

        JsonElement[] items = root.GetProperty("items").EnumerateArray().ToArray();
        Assert.True(items.Length >= 15);

        foreach (JsonElement item in items)
        {
            Assert.False(item.GetProperty("isRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.True(item.TryGetProperty("currentStatus", out _));
            Assert.True(item.TryGetProperty("cliSupported", out _));
            Assert.True(item.TryGetProperty("winFormsSupported", out _));
            Assert.True(item.GetProperty("nextImplementationPaths").GetArrayLength() >= 1);
        }

        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "load-engine" && item.GetProperty("currentStatus").GetString() == "bounded-runtime-output" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "bounded-benchmark-scheduler" && item.GetProperty("currentStatus").GetString() == "implemented-bounded-runtime" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "workspace-memory-pool" && item.GetProperty("currentStatus").GetString() == "implemented-readback-report");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "int8" && item.GetProperty("currentStatus").GetString() == "parse-report-only-calibration-boundary" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "timing-cache" && item.GetProperty("currentStatus").GetString() == "implemented-build-cache-lifecycle" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "plugin-library-boundary" && item.GetProperty("currentStatus").GetString() == "diagnostic-gui-cli" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "binding-metadata" &&
            item.GetProperty("currentStatus").GetString() == "implemented-pointer-free-multi-input-binding-multi-output-artifacts-and-reference-validation" &&
            item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "winforms-command-surface" && item.GetProperty("currentStatus").GetString() == "checklist-backed-command-preview" && item.GetProperty("winFormsSupported").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "package-consumer-runtime-proof-boundary" &&
            item.GetProperty("currentStatus").GetString() == "local-refitted-plan-package-consumer-runtime-public-proof-owner-action-required" &&
            !item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean() &&
            item.GetProperty("nextImplementationPaths").EnumerateArray().Any(path => path.GetString() == "samples/RefittedPlan.PackageConsumer"));
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "dynamic-shape" && item.GetProperty("nextImplementationPaths").EnumerateArray().Any(path => path.GetString() == "samples/YoloVision/yolovision-task-output-contract.json"));
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "binding-metadata" && item.GetProperty("nextImplementationPaths").EnumerateArray().Any(path => path.GetString() == "samples/YoloVision/yolovision-task-output-contract.json"));

        string markdown = File.ReadAllText(markdownPath);
        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("build-only report 不是 runtime proof", markdown, StringComparison.Ordinal);
        Assert.Contains("local feed、ProjectReference、direct `.nupkg` 不是 package-consumer-runtime proof", markdown, StringComparison.Ordinal);
        Assert.Contains("load-engine", markdown, StringComparison.Ordinal);
        Assert.Contains("bounded-runtime-output", markdown, StringComparison.Ordinal);
        Assert.Contains("bounded-benchmark-scheduler", markdown, StringComparison.Ordinal);
        Assert.Contains("workspace-memory-pool", markdown, StringComparison.Ordinal);
        Assert.Contains("implemented-readback-report", markdown, StringComparison.Ordinal);
        Assert.Contains("winforms-command-surface", markdown, StringComparison.Ordinal);
        Assert.Contains("checklist-backed-command-preview", markdown, StringComparison.Ordinal);
        Assert.Contains("implemented-pointer-free-multi-input-binding-multi-output-artifacts-and-reference-validation", markdown, StringComparison.Ordinal);
        Assert.Contains("samples/YoloVision/yolovision-task-output-contract.json", markdown, StringComparison.Ordinal);

        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime proof", article, StringComparison.Ordinal);
        Assert.Contains("WinForms parity", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrtexec-release-candidate-gap-list.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrtexec-release-candidate-gap-list.md", docsToc, StringComparison.Ordinal);
    }
}
