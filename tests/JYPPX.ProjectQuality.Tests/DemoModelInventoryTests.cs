using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DemoModelInventoryTests
{
    [Fact]
    public void InventoryCoversEveryFirstReleaseDeepLearningDemoModel()
    {
        using JsonDocument document = LoadInventory();
        JsonElement root = document.RootElement;
        JsonElement models = root.GetProperty("models");

        Assert.Equal("demo-model-acquisition-and-onnx-inventory", root.GetProperty("recordKind").GetString());
        Assert.Equal("E:/GitSpace/TensorRT-CSharp-API-4.0/models", root.GetProperty("workspaceModelRoot").GetString());
        Assert.Equal("eng/Sync-DemoOnnxModels.ps1", root.GetProperty("materializationScript").GetString());
        Assert.Equal(10, models.GetArrayLength());

        string[] expectedIds =
        {
            "classification-resnet18-imagenet1k-v1",
            "onnxtoengine-nvidia-mnist-opset8",
            "yolovision-yolov8n-detection-v8.3.0",
            "yolovision-yolov10n-detection-v1.1",
            "yolovision-yolox-s-detection-0.1.1rc0",
            "yolovision-yolov8n-classification-v8.3.0",
            "yolovision-yolov8n-instance-segmentation-v8.3.0",
            "yolovision-yolov8n-pose-v8.3.0",
            "yolovision-yolov8n-obb-v8.3.0",
            "yolovision-lraspp-mobilenet-v3-large-v0.25.0"
        };
        Assert.Equal(expectedIds, models.EnumerateArray().Select(model => model.GetProperty("id").GetString()));
    }

    [Fact]
    public void EveryInventoryEntryPinsAcquisitionConversionHashAndExternalStorage()
    {
        using JsonDocument document = LoadInventory();
        JsonElement root = document.RootElement;
        JsonElement policy = root.GetProperty("policy");

        Assert.True(policy.GetProperty("modelRootOutsideGitRepository").GetBoolean());
        Assert.False(policy.GetProperty("onnxFilesTrackedByGit").GetBoolean());
        Assert.False(policy.GetProperty("uploadsModelFiles").GetBoolean());
        Assert.False(policy.GetProperty("publishesModelFiles").GetBoolean());
        Assert.False(policy.GetProperty("cudaCudnnTensorRtNvrtcBundled").GetBoolean());

        foreach (JsonElement model in root.GetProperty("models").EnumerateArray())
        {
            JsonElement acquisition = model.GetProperty("acquisition");
            JsonElement conversion = model.GetProperty("conversion");
            JsonElement onnx = model.GetProperty("onnx");
            string path = Assert.IsType<string>(onnx.GetProperty("workspaceRelativePath").GetString());
            string sha256 = Assert.IsType<string>(onnx.GetProperty("sha256").GetString());

            Assert.False(string.IsNullOrWhiteSpace(acquisition.GetProperty("kind").GetString()));
            Assert.StartsWith("https://", acquisition.GetProperty("sourceUrl").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(acquisition.GetProperty("pinnedRevision").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(acquisition.GetProperty("script").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(conversion.GetProperty("kind").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(conversion.GetProperty("command").GetString()));
            Assert.StartsWith("models/", path, StringComparison.Ordinal);
            Assert.EndsWith(".onnx", path, StringComparison.Ordinal);
            Assert.True(onnx.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Matches("^[a-f0-9]{64}$", sha256);
            Assert.False(onnx.GetProperty("trackedByGit").GetBoolean());
        }
    }

    [Fact]
    public void EveryInventoryEntryLinksToArticlesThatExistAndCatalogNamesEveryModel()
    {
        using JsonDocument document = LoadInventory();
        JsonElement models = document.RootElement.GetProperty("models");
        string catalogPath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "demo-model-acquisition-and-onnx-conversion.md");
        string catalog = File.ReadAllText(catalogPath);
        var runtimeEvidencePaths = new HashSet<string>(StringComparer.Ordinal);

        foreach (JsonElement model in models.EnumerateArray())
        {
            string id = Assert.IsType<string>(model.GetProperty("id").GetString());
            JsonElement acquisition = model.GetProperty("acquisition");
            JsonElement conversion = model.GetProperty("conversion");
            JsonElement onnx = model.GetProperty("onnx");
            Assert.Contains($"`{id}`", catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(acquisition.GetProperty("sourceUrl").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(acquisition.GetProperty("pinnedRevision").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(acquisition.GetProperty("script").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(conversion.GetProperty("kind").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(conversion.GetProperty("command").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(conversion.GetProperty("toolchain").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(onnx.GetProperty("workspaceRelativePath").GetString()), catalog, StringComparison.Ordinal);
            Assert.Contains(Assert.IsType<string>(onnx.GetProperty("sha256").GetString()), catalog, StringComparison.Ordinal);
            foreach (JsonElement article in model.GetProperty("articles").EnumerateArray())
            {
                string relativePath = Assert.IsType<string>(article.GetString());
                Assert.True(File.Exists(Path.Combine(
                    RepositoryPaths.Root,
                    relativePath.Replace('/', Path.DirectorySeparatorChar))), relativePath);
            }

            string runtimeEvidence = Assert.IsType<string>(model.GetProperty("runtimeEvidence").GetString());
            string runtimeEvidencePath = Path.Combine(
                RepositoryPaths.Root,
                runtimeEvidence.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(runtimeEvidencePath), runtimeEvidence);
            using JsonDocument runtimeEvidenceDocument = JsonDocument.Parse(File.ReadAllText(runtimeEvidencePath));
            AssertRuntimeEvidenceIsPositiveAndNonPublishing(runtimeEvidence, runtimeEvidenceDocument.RootElement);
            Assert.True(runtimeEvidencePaths.Add(runtimeEvidence), runtimeEvidence);
        }

        Assert.Equal(models.GetArrayLength(), runtimeEvidencePaths.Count);

        Assert.Contains("不得把模型塞进 managed/native NuGet 包", catalog, StringComparison.Ordinal);
        Assert.Contains("真实运行证据索引", catalog, StringComparison.Ordinal);
        Assert.Contains("CUDA、cuDNN、TensorRT 与 NVRTC 始终由用户自行安装", catalog, StringComparison.Ordinal);

        string syncScript = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Sync-DemoOnnxModels.ps1"));
        Assert.Contains("demo-model-inventory.json", syncScript, StringComparison.Ordinal);
        Assert.Contains("modelRootOutsideGitRepository = $true", syncScript, StringComparison.Ordinal);
        Assert.Contains("uploadsAssets = $false", syncScript, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", syncScript, StringComparison.Ordinal);
    }

    private static void AssertRuntimeEvidenceIsPositiveAndNonPublishing(string path, JsonElement evidence)
    {
        Assert.Equal(1, evidence.GetProperty("schemaVersion").GetInt32());
        Assert.DoesNotContain(".template.", path, StringComparison.OrdinalIgnoreCase);

        string recordKind = Assert.IsType<string>(evidence.GetProperty("recordKind").GetString());
        string proofClassification = Assert.IsType<string>(evidence.GetProperty("proofClassification").GetString());
        Assert.Contains(proofClassification, new[] { "real-model-runtime", "source-tree-real-model-runtime" });

        if (recordKind == "sample-run-evidence-record")
        {
            Assert.Equal("real-model-runtime", proofClassification);
            Assert.False(evidence.GetProperty("templateOnly").GetBoolean());
            Assert.True(evidence.GetProperty("isSmokePassed").GetBoolean());
            Assert.True(evidence.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.Equal("real-model-runtime", evidence.GetProperty("validatorState").GetString());
            if (evidence.TryGetProperty("canPromotePackageConsumerRuntime", out JsonElement packagePromotion))
            {
                Assert.False(packagePromotion.GetBoolean());
            }

            AssertNonPublishingBoundary(evidence.GetProperty("proofBoundary"));
            return;
        }

        if (recordKind == "onnxtoengine-mnist-real-model-runtime-evidence")
        {
            JsonElement runtime = evidence.GetProperty("runtimeValidation");
            Assert.Equal(0, runtime.GetProperty("exitCode").GetInt32());
            Assert.True(runtime.GetProperty("inferenceRan").GetBoolean());
            Assert.True(runtime.GetProperty("outputMatch").GetBoolean());
            Assert.True(runtime.GetProperty("isRealModelRuntimeProof").GetBoolean());
            Assert.True(runtime.GetProperty("passed").GetBoolean());

            JsonElement negative = evidence.GetProperty("controlledNegativeValidation");
            Assert.NotEqual(0, negative.GetProperty("exitCode").GetInt32());
            Assert.False(negative.GetProperty("outputMatch").GetBoolean());
            Assert.False(negative.GetProperty("isRealModelRuntimeProof").GetBoolean());
            Assert.True(negative.GetProperty("failClosed").GetBoolean());
            AssertNonPublishingBoundary(evidence.GetProperty("proofBoundary"));
            return;
        }

        Assert.Contains(recordKind, new[]
        {
            "yolov10-official-source-tree-runtime-proof-closure",
            "yolox-official-source-tree-runtime-proof-closure"
        });
        Assert.Equal("source-tree-real-model-runtime", proofClassification);
        Assert.Equal("passed", evidence.GetProperty("engineBuild").GetProperty("status").GetString());
        JsonElement closureRuntime = evidence.GetProperty("runtime");
        Assert.Equal("passed", closureRuntime.GetProperty("status").GetString());
        Assert.Equal("YoloVision Passed=True", closureRuntime.GetProperty("passedMarker").GetString());
        Assert.True(closureRuntime.GetProperty("predictionCount").GetInt32() > 0);
        AssertNonPublishingBoundary(evidence.GetProperty("boundary"));
    }

    private static void AssertNonPublishingBoundary(JsonElement boundary)
    {
        Assert.Equal(JsonValueKind.Object, boundary.ValueKind);
        foreach (string propertyName in new[]
        {
            "packageConsumerRuntimeProof",
            "isPackageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "isPostPublishProof",
            "ownerReleaseAcceptance",
            "canPublishPublicly",
            "releaseProof",
            "canCloseReleaseIssue",
            "performsPublish",
            "uploadsAssets"
        })
        {
            if (boundary.TryGetProperty(propertyName, out JsonElement value))
            {
                Assert.False(value.GetBoolean());
            }
        }
    }

    private static JsonDocument LoadInventory()
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "demo-model-inventory.json")));
    }
}
