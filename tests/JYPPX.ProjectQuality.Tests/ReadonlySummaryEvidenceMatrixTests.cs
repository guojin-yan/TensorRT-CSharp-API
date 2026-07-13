using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReadonlySummaryEvidenceMatrixTests
{
    private static readonly string[] RequiredIds =
    {
        "tensorrt-runtime-diagnostic-summary",
        "tensorrt-engine-deployment-summary",
        "tensorrt-builder-config-deployment-summary",
        "tensorrt-execution-context-deployment-summary",
        "tensorrt-serialization-config-summary",
        "tensorrt-runtime-config-summary",
        "tensorrt-onnx-parser-diagnostic-summary",
        "tensorrt-onnx-model-support-summary",
        "tensorrt-error-recorder-summary",
        "tensorrt-plugin-creator-v3-metadata-design-gate",
        "tensorrt-plugin-registry-inventory-summary",
        "cuda-graph-diagnostic-summary",
        "cuda-graph-exec-diagnostic-summary",
        "cuda-device-graph-memory-summary",
        "cuda-memory-range-diagnostic-summary"
    };

    [Fact]
    public void MatrixCoversReadonlySummariesWithoutPromotingRuntimeProof()
    {
        JsonElement root = ReadMatrix();
        JsonElement boundary = root.GetProperty("boundary");

        Assert.False(boundary.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPostPublishVerification").GetBoolean());
        Assert.False(boundary.GetProperty("canDeleteDeferredRecord").GetBoolean());

        string matrixText = File.ReadAllText(MatrixPath());
        Assert.Contains("readonly diagnostics", matrixText);
        Assert.Contains("readonly summary", matrixText);
        Assert.Contains("TensorRtExec report", matrixText);
        Assert.Contains("YoloVision matrix", matrixText);
        Assert.Contains("OnnxToEngine report", matrixText);
        Assert.Contains("not-runtime-proof", matrixText);
        Assert.Contains("Clean package-consumer runtime smoke", matrixText, StringComparison.OrdinalIgnoreCase);

        JsonElement entries = root.GetProperty("entries");
        Assert.True(entries.GetArrayLength() >= RequiredIds.Length);

        foreach (string id in RequiredIds)
        {
            JsonElement entry = FindEntry(entries, id);

            Assert.Equal(id, entry.GetProperty("id").GetString());
            Assert.False(entry.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(entry.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.True(entry.GetProperty("pointerFreePublicSurface").GetBoolean());
            Assert.Equal("not-runtime-proof", entry.GetProperty("runtimeEvidenceKind").GetString());
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("publicType").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("source").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("entryPoint").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("notRuntimeProofReason").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(entry.GetProperty("nextProofRequired").GetString()));
            Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, entry.GetProperty("source").GetString()!)), id);

            JsonElement tests = entry.GetProperty("qualityTests");
            Assert.True(tests.GetArrayLength() >= 1, id);
            foreach (JsonElement testPath in tests.EnumerateArray())
            {
                Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, testPath.GetString()!)), $"{id}:{testPath.GetString()}");
            }
        }
    }

    [Fact]
    public void MatrixMarkersAreBackedBySmokeOrQualityEvidence()
    {
        string tensorRtSmoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");
        string cudaGraphSmoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string cudaSmoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string callbackSmoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");
        string pluginInventorySmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string allSmokeEvidence = string.Concat(tensorRtSmoke, cudaGraphSmoke, cudaSmoke, callbackSmoke, onnxSmoke, pluginInventorySmoke, smokeReadme);

        foreach (JsonElement entry in ReadMatrix().GetProperty("entries").EnumerateArray())
        {
            string id = entry.GetProperty("id").GetString()!;
            string marker = entry.GetProperty("smokeMarker").GetString()!;
            Assert.Contains(marker, allSmokeEvidence);

            foreach (JsonElement testPath in entry.GetProperty("qualityTests").EnumerateArray())
            {
                string testSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, testPath.GetString()!));
                if (testSource.Contains(marker, StringComparison.Ordinal) ||
                    testSource.Contains(entry.GetProperty("publicType").GetString()!, StringComparison.Ordinal) ||
                    testSource.Contains(entry.GetProperty("entryPoint").GetString()!, StringComparison.Ordinal))
                {
                    goto NextEntry;
                }
            }

            throw new InvalidOperationException("No quality test names marker/type/entry point for " + id);

        NextEntry:
            continue;
        }
    }

    [Fact]
    public void PublicSummarySourcesRemainPointerFreeAndSelfBounded()
    {
        foreach (JsonElement entry in ReadMatrix().GetProperty("entries").EnumerateArray())
        {
            string source = File.ReadAllText(Path.Combine(RepositoryPaths.Root, entry.GetProperty("source").GetString()!));
            string publicType = entry.GetProperty("publicType").GetString()!;

            Assert.Contains(publicType, source);
            Assert.Contains("CanPromoteRuntimeProof", source);
            Assert.Contains("false", source);
            Assert.DoesNotContain("public IntPtr", source);
            Assert.DoesNotContain("public nint", source);
        }
    }

    [Fact]
    public void DocumentationStatesSummaryMatrixIsNotRuntimeProof()
    {
        string matrixDoc = ReadSource("docs", "articles", "zh-cn", "readonly-summary-evidence-matrix.md");
        string readonlyPlaybook = ReadSource("docs", "articles", "zh-cn", "deferred-readonly-api-upgrade-playbook.md");
        string packageConsumerPlaybook = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-playbook.md");

        Assert.Contains("非 runtime proof", matrixDoc);
        Assert.Contains("package-consumer-runtime", matrixDoc);
        Assert.Contains("post-publish verification", matrixDoc);
        Assert.Contains("clean consumer restore", matrixDoc);
        Assert.Contains("strict validator", matrixDoc);
        Assert.Contains("readonly diagnostics", readonlyPlaybook);
        Assert.Contains("不能", readonlyPlaybook);
        Assert.Contains("不是 runtime proof", packageConsumerPlaybook);
    }

    private static JsonElement FindEntry(JsonElement entries, string id)
    {
        foreach (JsonElement entry in entries.EnumerateArray())
        {
            if (entry.GetProperty("id").GetString() == id)
            {
                return entry;
            }
        }

        throw new InvalidOperationException("Matrix entry not found: " + id);
    }

    private static JsonElement ReadMatrix()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(MatrixPath()));
        return document.RootElement.Clone();
    }

    private static string MatrixPath()
    {
        return Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "readonly-summary-evidence-matrix.json");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
