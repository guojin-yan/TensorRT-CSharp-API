using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecParityMatrixArtifactTests
{
    [Fact]
    public void TensorRtExecParityMatrixArtifactIsMachineReadableAndKeepsProofBoundary()
    {
        string jsonPath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.md");
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrt-exec-trtexec-parity-matrix.md");
        string commandSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs"));
        string appReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));

        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("tensor-rt-exec-trtexec-parity-matrix", root.GetProperty("matrixId").GetString());
        Assert.Contains("not runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] entries = root.GetProperty("entries").EnumerateArray().ToArray();
        Assert.True(entries.Length >= 19);

        string[] requiredIds =
        {
            "onnx-input",
            "save-engine",
            "load-engine",
            "dynamic-shape",
            "shape-profile",
            "shape-alias-batch",
            "timing-iterations",
            "yolovision-owner-backfill-shape-profiles",
            "fp16",
            "int8",
            "precision-shortcuts-debug-boundary",
            "workspace-memory-pool",
            "timing-cache",
            "gui-cli-field-map",
            "plugin-library-boundary",
            "bounded-benchmark-scheduler",
            "profiling",
            "wait-idle-controls",
            "layer-dump",
            "report-export-alias",
            "verbose-logging",
            "binding-metadata",
            "package-consumer-runtime-proof-boundary"
        };

        foreach (string id in requiredIds)
        {
            JsonElement entry = Assert.Single(entries, item => item.GetProperty("id").GetString() == id);
            Assert.False(entry.GetProperty("isRuntimeProof").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("trtexecOption").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("tensorRtExecStatus").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("proofBoundary").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("gap").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("nextStep").GetString()));
            Assert.True(entry.GetProperty("entryPoints").GetArrayLength() >= 1);
        }

        string markdown = File.ReadAllText(markdownPath);
        string article = File.ReadAllText(articlePath);
        foreach (string id in requiredIds)
        {
            Assert.Contains(id, markdown, StringComparison.Ordinal);
        }

        foreach (string option in new[]
        {
            "--onnx",
            "--saveEngine",
            "--loadEngine",
            "--shapes",
            "--inputShapes",
            "--batch",
            "--fp16",
            "--int8",
            "--fp8",
            "--best",
            "--dumpRefit",
            "--allowWeightStreaming",
            "--markDebug",
            "--dumpDebugTensors",
            "--workspace",
            "--memPoolSize",
            "--timingCacheFile",
            "--exportTimingCache",
            "--calib",
            "--plugins",
            "--dynamicPlugins",
            "--setPluginsToSerialize",
            "--sleepTime",
            "--idleTime",
            "--exportProfile",
            "--exportLayerInfo",
            "--exportReport",
            "--report",
            "--dumpRawBindingsToFile"
        })
        {
            Assert.Contains(option, article + commandSource + appReadme, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("package-consumer-runtime", markdown, StringComparison.Ordinal);
        Assert.Contains("does not allow `TensorRtExec` itself to declare release proof", markdown, StringComparison.Ordinal);
        Assert.Contains("gui-cli-field-map", markdown, StringComparison.Ordinal);
        Assert.Contains("parse/report-only", markdown + article + appReadme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("OptionImplementationStatus", article + appReadme, StringComparison.Ordinal);
    }
}
