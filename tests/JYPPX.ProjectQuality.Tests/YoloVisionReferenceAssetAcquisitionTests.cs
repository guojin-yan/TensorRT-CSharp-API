using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionReferenceAssetAcquisitionTests
{
    [Fact]
    public void YoloV8PoseOfficialManifestPinsEmbeddedChannelsAndHumanInput()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-pose-official-assets.json");
        string scriptPath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Acquire-YoloV8PoseOfficialAssets.ps1");
        Assert.True(File.Exists(manifestPath), manifestPath);
        Assert.True(File.Exists(scriptPath), scriptPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("yolovision-yolov8n-pose-official-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal(177482232, root.GetProperty("upstreamReleaseId").GetInt64());
        Assert.Equal("6e43d1e1e5db72afbf686dee6745669bcb124b0a", root.GetProperty("upstreamSourceCommit").GetString());
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.False(root.GetProperty("license").GetProperty("publicRedistributionOwnerApproval").GetBoolean());

        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 56, 8400 }, output.GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal("detection-rows-with-embedded-pose-keypoints", output.GetProperty("role").GetString());
        Assert.Equal(5, output.GetProperty("auxiliaryChannelStart").GetInt32());
        Assert.Equal(17, output.GetProperty("keypointCount").GetInt32());
        Assert.Equal(3, output.GetProperty("keypointStride").GetInt32());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(3, assets.Length);
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
        });
        JsonElement weights = assets.Single(static asset => asset.GetProperty("id").GetString() == "yolov8n-pose-pt");
        Assert.Equal(195719300, weights.GetProperty("githubReleaseAssetId").GetInt64());
        JsonElement image = assets.Single(static asset => asset.GetProperty("id").GetString() == "ultralytics-bus-jpg");
        Assert.Contains(root.GetProperty("upstreamSourceCommit").GetString()!, image.GetProperty("url").GetString()!, StringComparison.Ordinal);

        JsonElement derived = root.GetProperty("derivedInput");
        Assert.Equal("P6 RGB PPM", derived.GetProperty("format").GetString());
        Assert.Equal(810, derived.GetProperty("width").GetInt32());
        Assert.Equal(1080, derived.GetProperty("height").GetInt32());
        Assert.Equal(64, derived.GetProperty("expectedSha256").GetString()!.Length);

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("Test-DriveIsNotC", script, StringComparison.Ordinal);
        Assert.Contains("JYPPX_YOLO_PYTHON", script, StringComparison.Ordinal);
        Assert.Contains("expectedSha256", script, StringComparison.Ordinal);
        Assert.Contains("upstream Release did not publish a digest", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("performsExport = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloV8PoseRealRuntimeEvidenceProvesEmbeddedDecodeAndKeepsReleaseBoundariesFalse()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-pose-real-model-runtime-evidence.json");
        Assert.True(File.Exists(evidencePath), evidencePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        Assert.Equal("sample-run-evidence-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("templateOnly").GetBoolean());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899", root.GetProperty("modelSha256").GetString());

        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 56, 8400 }, output.GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal(5, output.GetProperty("auxiliaryChannelStart").GetInt32());
        Assert.Equal(17, output.GetProperty("keypointCount").GetInt32());
        Assert.Equal(3, output.GetProperty("keypointStride").GetInt32());

        JsonElement raw = root.GetProperty("runtimeReferenceValidation");
        Assert.True(raw.GetProperty("passed").GetBoolean());
        JsonElement rawTensor = Assert.Single(raw.GetProperty("tensors").EnumerateArray());
        Assert.Equal(470_400, rawTensor.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, rawTensor.GetProperty("mismatchCount").GetInt32());

        JsonElement pose = root.GetProperty("posePostprocessValidation");
        Assert.True(pose.GetProperty("passed").GetBoolean());
        Assert.Equal(4, pose.GetProperty("predictionCount").GetInt32());
        Assert.True(pose.GetProperty("minimumObservedBoxIoU").GetDouble() >= pose.GetProperty("thresholds").GetProperty("minimumBoxIoU").GetDouble());
        Assert.True(pose.GetProperty("maximumObservedKeypointCoordinateError").GetDouble() <= pose.GetProperty("thresholds").GetProperty("maximumKeypointCoordinateError").GetDouble());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt32());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());
        Assert.False(negative.GetProperty("passed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("sourceTreeRealModelRuntime").GetBoolean());
        foreach (string name in new[]
        {
            "publicRedistributionApproved",
            "packageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "ownerReleaseAcceptance",
            "releaseProof",
            "performsPublish",
            "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean(), name);
        }

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-pose-tutorial.md"));
        foreach (string text in new[] { readme, tutorial })
        {
            Assert.Contains("[1,56,8400]", text, StringComparison.Ordinal);
            Assert.Contains("--aux-channel-start", text, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", text, StringComparison.Ordinal);
            Assert.Contains("package-consumer", text, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void YoloV8ObbOfficialManifestPinsEmbeddedAngleRotatedNmsAndEdriveAcquisition()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-obb-official-assets.json");
        string acquisitionPath = Path.Combine(RepositoryPaths.Root, "eng", "Acquire-YoloV8ObbOfficialAssets.ps1");
        string referencePath = Path.Combine(RepositoryPaths.Root, "eng", "Invoke-YoloVisionObbReference.py");
        Assert.True(File.Exists(manifestPath), manifestPath);
        Assert.True(File.Exists(acquisitionPath), acquisitionPath);
        Assert.True(File.Exists(referencePath), referencePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("yolovision-yolov8n-obb-official-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal(177482232, root.GetProperty("upstreamReleaseId").GetInt64());
        Assert.Equal("6e43d1e1e5db72afbf686dee6745669bcb124b0a", root.GetProperty("upstreamSourceCommit").GetString());
        Assert.Equal("428939c3e501f70ab2a0ded889663efcf5bfbe6c", root.GetProperty("upstreamImageCommit").GetString());
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.False(root.GetProperty("license").GetProperty("publicRedistributionOwnerApproval").GetBoolean());

        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 20, 21504 }, output.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal("detection-rows-with-embedded-obb-angle", output.GetProperty("role").GetString());
        Assert.Equal(15, output.GetProperty("classCount").GetInt32());
        Assert.Equal(19, output.GetProperty("auxiliaryChannelStart").GetInt32());
        Assert.Equal("radians", output.GetProperty("angleUnit").GetString());
        Assert.Contains("probabilistic-iou", output.GetProperty("nms").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(3, assets.Length);
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
        });
        JsonElement weights = assets.Single(static asset => asset.GetProperty("id").GetString() == "yolov8n-obb-pt");
        Assert.Equal(195719551, weights.GetProperty("githubReleaseAssetId").GetInt64());
        Assert.Equal(1920, root.GetProperty("derivedInput").GetProperty("width").GetInt32());
        Assert.Equal(1080, root.GetProperty("derivedInput").GetProperty("height").GetInt32());
        Assert.Equal(15, output.GetProperty("classNames").GetArrayLength());

        string acquisition = File.ReadAllText(acquisitionPath);
        Assert.Contains("Test-DriveIsNotC", acquisition, StringComparison.Ordinal);
        Assert.Contains("expectedSha256", acquisition, StringComparison.Ordinal);
        Assert.Contains("derivedLabels", acquisition, StringComparison.Ordinal);
        Assert.Contains("performsRuntime = $false", acquisition, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", acquisition, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", acquisition, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", acquisition, StringComparison.OrdinalIgnoreCase);

        string reference = File.ReadAllText(referencePath);
        Assert.Contains("CPUExecutionProvider", reference, StringComparison.Ordinal);
        Assert.Contains("rotatedRectangleIntersection", reference, StringComparison.Ordinal);
        Assert.Contains("periodic_angle_error", reference, StringComparison.Ordinal);
        Assert.Contains("minimum-rotated-iou", reference, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloV8ClassificationOfficialManifestPinsSoftmaxLabelsAndCenterCropAcquisition()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-cls-official-assets.json");
        string acquisitionPath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Acquire-YoloV8ClassificationOfficialAssets.ps1");
        string referencePath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-YoloVisionClassificationReference.py");
        Assert.True(File.Exists(manifestPath), manifestPath);
        Assert.True(File.Exists(acquisitionPath), acquisitionPath);
        Assert.True(File.Exists(referencePath), referencePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal("yolovision-yolov8n-cls-official-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal(177482232, root.GetProperty("upstreamReleaseId").GetInt64());
        Assert.Equal("6e43d1e1e5db72afbf686dee6745669bcb124b0a", root.GetProperty("upstreamSourceCommit").GetString());
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.False(root.GetProperty("license").GetProperty("publicRedistributionOwnerApproval").GetBoolean());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(4, assets.Length);
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
        });
        JsonElement weights = assets.Single(static asset => asset.GetProperty("id").GetString() == "yolov8n-cls-pt");
        Assert.Equal(195719213, weights.GetProperty("githubReleaseAssetId").GetInt64());

        JsonElement labels = root.GetProperty("derivedLabels");
        Assert.Equal(1000, labels.GetProperty("classCount").GetInt32());
        Assert.Contains("map", labels.GetProperty("derivation").GetString(), StringComparison.Ordinal);
        Assert.Contains("not", labels.GetProperty("importantBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement input = root.GetProperty("modelContract").GetProperty("input");
        Assert.Equal(new[] { 1, 3, 224, 224 }, input.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 1000 }, output.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal("probabilities", output.GetProperty("role").GetString());
        Assert.Equal("Softmax", output.GetProperty("lastOnnxNode").GetString());
        Assert.Equal("probabilities", root.GetProperty("modelContract").GetProperty("postprocess").GetProperty("classificationScoreMode").GetString());

        string acquisition = File.ReadAllText(acquisitionPath);
        Assert.Contains("Test-DriveIsNotC", acquisition, StringComparison.Ordinal);
        Assert.Contains("expectedSha256", acquisition, StringComparison.Ordinal);
        Assert.Contains("performsExport = $false", acquisition, StringComparison.Ordinal);
        Assert.Contains("performsRuntime = $false", acquisition, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", acquisition, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", acquisition, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", acquisition, StringComparison.OrdinalIgnoreCase);

        string reference = File.ReadAllText(referencePath);
        Assert.Contains("CPUExecutionProvider", reference, StringComparison.Ordinal);
        Assert.Contains("lastOnnxNode", reference, StringComparison.Ordinal);
        Assert.Contains("output0.tampered.reference.json", reference, StringComparison.Ordinal);
        Assert.Contains("sameTop5IndicesAndOrder", reference, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloV8ClassificationRealRuntimeEvidenceProvesFullVectorTop5AndNegativeReference()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-cls-real-model-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("templateOnly").GetBoolean());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());

        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 1000 }, output.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal("probabilities", output.GetProperty("role").GetString());
        Assert.Equal("Softmax", output.GetProperty("lastOnnxNode").GetString());
        Assert.False(root.GetProperty("modelContract").GetProperty("postprocess").GetProperty("applyNms").GetBoolean());

        JsonElement independent = root.GetProperty("independentReferenceValidation");
        Assert.True(independent.GetProperty("passed").GetBoolean());
        Assert.True(independent.GetProperty("pytorchOnnxRuntimeMaximumAbsoluteError").GetDouble() < 1e-5);
        Assert.True(root.GetProperty("csharpCenterCropValidation").GetProperty("sameTop5IndicesAndOrder").GetBoolean());

        JsonElement raw = root.GetProperty("runtimeReferenceValidation");
        Assert.True(raw.GetProperty("passed").GetBoolean());
        JsonElement tensor = Assert.Single(raw.GetProperty("tensors").EnumerateArray());
        Assert.Equal(1000, tensor.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, tensor.GetProperty("mismatchCount").GetInt32());
        Assert.True(tensor.GetProperty("maximumAbsoluteError").GetDouble() <= raw.GetProperty("absoluteTolerance").GetDouble());

        JsonElement top5 = root.GetProperty("top5Validation");
        Assert.True(top5.GetProperty("passed").GetBoolean());
        Assert.Equal(new[] { 654, 734, 874, 575, 612 }, top5.GetProperty("predictions").EnumerateArray().Select(static item => item.GetProperty("classIndex").GetInt32()).ToArray());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt32());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("sourceTreeRealModelRuntime").GetBoolean());
        foreach (string name in new[]
        {
            "publicRedistributionApproved",
            "packageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "ownerReleaseAcceptance",
            "releaseProof",
            "performsPublish",
            "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean(), name);
        }

        string tutorial = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-classification-yolov8n-labels-topk-guide.md"));
        Assert.Contains("[1,1000]", tutorial, StringComparison.Ordinal);
        Assert.Contains("Mismatches=1", tutorial, StringComparison.Ordinal);
        Assert.Contains("classification-score-mode probabilities", tutorial, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", tutorial, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloV8ObbRealRuntimeEvidenceProvesRawAndRotatedGeometryComparisons()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-obb-real-model-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("templateOnly").GetBoolean());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92", root.GetProperty("modelSha256").GetString());

        JsonElement output = Assert.Single(root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray());
        Assert.Equal(new[] { 1, 20, 21504 }, output.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal(19, output.GetProperty("auxiliaryChannelStart").GetInt32());
        Assert.Contains("probabilistic-iou", output.GetProperty("nms").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement raw = root.GetProperty("runtimeReferenceValidation");
        Assert.True(raw.GetProperty("passed").GetBoolean());
        JsonElement tensor = Assert.Single(raw.GetProperty("tensors").EnumerateArray());
        Assert.Equal(430_080, tensor.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, tensor.GetProperty("mismatchCount").GetInt32());

        JsonElement obb = root.GetProperty("obbPostprocessValidation");
        Assert.True(obb.GetProperty("passed").GetBoolean());
        Assert.Equal(40, obb.GetProperty("predictionCount").GetInt32());
        Assert.True(obb.GetProperty("minimumObservedRotatedIoU").GetDouble() >= obb.GetProperty("thresholds").GetProperty("minimumRotatedIoU").GetDouble());
        Assert.True(obb.GetProperty("maximumObservedAngleErrorRadians").GetDouble() <= obb.GetProperty("thresholds").GetProperty("maximumAngleErrorRadians").GetDouble());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt32());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("sourceTreeRealModelRuntime").GetBoolean());
        foreach (string name in new[]
        {
            "publicRedistributionApproved",
            "packageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "ownerReleaseAcceptance",
            "releaseProof",
            "performsPublish",
            "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean(), name);
        }

        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-obb-tutorial.md"));
        Assert.Contains("[1,20,21504]", tutorial, StringComparison.Ordinal);
        Assert.Contains("0.997781", tutorial, StringComparison.Ordinal);
        Assert.Contains("Mismatches=1", tutorial, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", tutorial, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloV8SegmentationOfficialManifestPinsReleaseAssetLicenseAndOutputRoles()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-seg-official-assets.json");
        string scriptPath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Acquire-YoloV8SegOfficialAssets.ps1");
        Assert.True(File.Exists(manifestPath), manifestPath);
        Assert.True(File.Exists(scriptPath), scriptPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal(
            "yolovision-yolov8n-seg-official-asset-acquisition-manifest",
            root.GetProperty("recordKind").GetString());
        Assert.Equal(177482232, root.GetProperty("upstreamReleaseId").GetInt64());
        Assert.Equal("v8.3.0", root.GetProperty("upstreamReleaseTag").GetString());
        Assert.Equal(40, root.GetProperty("upstreamSourceCommit").GetString()!.Length);
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.False(root.GetProperty("license").GetProperty("publicRedistributionOwnerApproval").GetBoolean());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(2, assets.Length);
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
            Assert.False(string.IsNullOrWhiteSpace(asset.GetProperty("hashProvenance").GetString()));
        });
        JsonElement modelAsset = assets.Single(static asset => asset.GetProperty("id").GetString() == "yolov8n-seg-pt");
        Assert.Equal(195720083, modelAsset.GetProperty("githubReleaseAssetId").GetInt64());
        Assert.Contains("upstream-release-api-digest-was-empty", modelAsset.GetProperty("hashProvenance").GetString(), StringComparison.Ordinal);

        JsonElement[] outputs = root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray().ToArray();
        Assert.Equal(new[] { "output0", "output1" }, outputs.Select(static output => output.GetProperty("name").GetString()).ToArray());
        Assert.Equal("detection-rows-with-mask-coefficients", outputs[0].GetProperty("role").GetString());
        Assert.Equal(new[] { 1, 116, 8400 }, outputs[0].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal(32, outputs[0].GetProperty("maskCoefficientCount").GetInt32());
        Assert.Equal("mask-prototypes", outputs[1].GetProperty("role").GetString());
        Assert.Equal(new[] { 1, 32, 160, 160 }, outputs[1].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("performsPublish").GetBoolean());

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("Test-DriveIsNotC", script, StringComparison.Ordinal);
        Assert.Contains("expectedLength", script, StringComparison.Ordinal);
        Assert.Contains("expectedSha256", script, StringComparison.Ordinal);
        Assert.Contains("upstream Release did not publish a digest", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("performsExport = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloV8SegmentationRealRuntimeEvidenceKeepsPromotionAndReleaseBoundariesSeparate()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-seg-real-model-runtime-evidence.json");
        Assert.True(File.Exists(evidencePath), evidencePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        Assert.Equal("sample-run-evidence-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("templateOnly").GetBoolean());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("real-model-runtime", root.GetProperty("validatorState").GetString());
        Assert.Equal(
            "08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d",
            root.GetProperty("modelSha256").GetString());

        JsonElement contract = root.GetProperty("modelContract");
        Assert.Equal(
            new[] { 1, 3, 640, 640 },
            contract.GetProperty("input").GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        JsonElement[] outputs = contract.GetProperty("outputs").EnumerateArray().ToArray();
        Assert.Equal(new[] { "output0", "output1" }, outputs.Select(static output => output.GetProperty("name").GetString()).ToArray());
        Assert.Equal(new[] { 1, 116, 8400 }, outputs[0].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal(new[] { 1, 32, 160, 160 }, outputs[1].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());

        JsonElement runtimeReference = root.GetProperty("runtimeReferenceValidation");
        Assert.True(runtimeReference.GetProperty("passed").GetBoolean());
        JsonElement[] referenceTensors = runtimeReference.GetProperty("tensors").EnumerateArray().ToArray();
        Assert.Equal(2, referenceTensors.Length);
        Assert.Equal(1_793_600, referenceTensors.Sum(static tensor => tensor.GetProperty("comparedValueCount").GetInt32()));
        Assert.All(referenceTensors, static tensor =>
        {
            Assert.Equal(0, tensor.GetProperty("mismatchCount").GetInt32());
            Assert.Equal(64, tensor.GetProperty("referenceSha256").GetString()!.Length);
        });

        JsonElement postprocess = root.GetProperty("segmentationPostprocessValidation");
        Assert.True(postprocess.GetProperty("passed").GetBoolean());
        JsonElement thresholds = postprocess.GetProperty("thresholds");
        JsonElement[] predictions = postprocess.GetProperty("predictions").EnumerateArray().ToArray();
        Assert.Equal(4, predictions.Length);
        Assert.Equal(new[] { "dog", "bicycle", "truck", "car" }, predictions.Select(static prediction => prediction.GetProperty("className").GetString()).ToArray());
        Assert.All(predictions, prediction =>
        {
            Assert.True(prediction.GetProperty("passed").GetBoolean());
            Assert.True(prediction.GetProperty("boxIoU").GetDouble() >= thresholds.GetProperty("minimumBoxIoU").GetDouble());
            Assert.True(prediction.GetProperty("maskIoU").GetDouble() >= thresholds.GetProperty("minimumMaskIoU").GetDouble());
        });

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(2, negative.GetProperty("exitCode").GetInt32());
        Assert.True(negative.GetProperty("completed").GetBoolean());
        Assert.False(negative.GetProperty("passed").GetBoolean());
        Assert.Equal(974_400, negative.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt32());

        JsonElement artifactIntegrity = root.GetProperty("controlledArtifactIntegrityValidation");
        Assert.True(artifactIntegrity.GetProperty("failClosed").GetBoolean());
        Assert.False(artifactIntegrity.GetProperty("passed").GetBoolean());
        Assert.Contains("SHA256 does not match", artifactIntegrity.GetProperty("diagnostic").GetString(), StringComparison.Ordinal);

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("sourceTreeRealModelRuntime").GetBoolean());
        foreach (string name in new[]
        {
            "publicRedistributionApproved",
            "packageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "ownerReleaseAcceptance",
            "releaseProof",
            "performsPublish",
            "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean(), name);
        }

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8-seg-real-asset-tutorial.md"));
        string maskGuide = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-segmentation-mask-postprocess-guide.md"));
        foreach (string text in new[] { readme, tutorial, maskGuide })
        {
            Assert.Contains("--segmentation-mask-output-directory", text, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", text, StringComparison.Ordinal);
            Assert.Contains("package", text, StringComparison.OrdinalIgnoreCase);
        }
        Assert.Contains("Mismatches=1", tutorial, StringComparison.Ordinal);
        Assert.Contains("mask IoU", tutorial, StringComparison.Ordinal);
        Assert.Contains("publicRedistributionApproved", File.ReadAllText(evidencePath), StringComparison.Ordinal);
    }

    [Fact]
    public void RepositoryManifestKeepsHashVerifiedTensorRtAssetsBehindLicenseReview()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-reference-assets.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-reference-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal("source-files-hash-verified-license-review-required", root.GetProperty("status").GetString());
        Assert.Equal("owner-review-required", root.GetProperty("licenseBoundary").GetProperty("state").GetString());
        Assert.False(root.GetProperty("licenseBoundary").GetProperty("allRedistributionApproved").GetBoolean());
        Assert.False(root.GetProperty("licenseBoundary").GetProperty("canPromoteRealModelRuntime").GetBoolean());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(3, assets.Length);
        Assert.Equal(new[] { "model", "labels", "input-image" }, assets.Select(static asset => asset.GetProperty("role").GetString()).ToArray());
        Assert.All(assets, static asset =>
        {
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal("owner-review-required", asset.GetProperty("license").GetProperty("status").GetString());
            Assert.False(asset.GetProperty("license").GetProperty("redistributionApproved").GetBoolean());
        });
    }

    [Fact]
    public void AcquisitionScriptVerifiesAndCopiesFilesWithoutPromotingUnreviewedLicenses()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-assets-" + Guid.NewGuid().ToString("N"));
        string sourceRoot = Path.Combine(tempRoot, "source");
        string outputRoot = Path.Combine(tempRoot, "output");
        Directory.CreateDirectory(Path.Combine(sourceRoot, "bin"));
        Directory.CreateDirectory(Path.Combine(sourceRoot, "samples"));

        try
        {
            byte[] model = Encoding.UTF8.GetBytes("model-fixture");
            byte[] labels = Encoding.UTF8.GetBytes("label-a\nlabel-b\n");
            File.WriteAllBytes(Path.Combine(sourceRoot, "bin", "model.onnx"), model);
            File.WriteAllBytes(Path.Combine(sourceRoot, "samples", "labels.txt"), labels);

            object manifest = new
            {
                schemaVersion = 1,
                recordKind = "yolovision-reference-asset-acquisition-manifest",
                assetSetId = "test-fixture",
                sourceRootHints = Array.Empty<string>(),
                assets = new object[]
                {
                    NewAsset("model", "model", @"bin\model.onnx", "model.onnx", model),
                    NewAsset("labels", "labels", @"samples\labels.txt", "labels.txt", labels),
                },
            };
            string manifestPath = Path.Combine(tempRoot, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));

            string output = RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Acquire-YoloVisionReferenceAssets.ps1"),
                "-ManifestPath",
                manifestPath,
                "-SourceRoot",
                sourceRoot,
                "-OutputRoot",
                outputRoot);

            Assert.Contains("AcquisitionState=verified-local-source-owner-review-required", output, StringComparison.Ordinal);
            string reportPath = Path.Combine(outputRoot, "acquisition-report.json");
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(reportPath));
            JsonElement report = reportDocument.RootElement;
            Assert.Equal("verified-local-source-owner-review-required", report.GetProperty("acquisitionState").GetString());
            Assert.True(report.GetProperty("allFilesReady").GetBoolean());
            Assert.False(report.GetProperty("allLicensesReady").GetBoolean());
            Assert.False(report.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(report.GetProperty("canRedistributeInRepository").GetBoolean());
            Assert.False(report.GetProperty("performsPublish").GetBoolean());
            Assert.Equal(2, report.GetProperty("assets").GetArrayLength());
            Assert.All(report.GetProperty("assets").EnumerateArray(), static asset =>
            {
                Assert.True(asset.GetProperty("fileReady").GetBoolean());
                Assert.True(asset.GetProperty("copied").GetBoolean());
                Assert.False(asset.GetProperty("licenseReady").GetBoolean());
            });

            Assert.Equal(model, File.ReadAllBytes(Path.Combine(outputRoot, "model.onnx")));
            Assert.Equal(labels, File.ReadAllBytes(Path.Combine(outputRoot, "labels.txt")));

            string script = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "eng",
                "Acquire-YoloVisionReferenceAssets.ps1"));
            Assert.Contains("AllowDownload", script, StringComparison.Ordinal);
            Assert.Contains("RequireLicenseReady", script, StringComparison.Ordinal);
            Assert.Contains("Invoke-WebRequest", script, StringComparison.Ordinal);
            Assert.Contains("canPromoteRealModelRuntime", script, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static object NewAsset(
        string id,
        string role,
        string relativeSourcePath,
        string cacheFileName,
        byte[] contents)
    {
        return new
        {
            id,
            role,
            relativeSourcePath,
            cacheFileName,
            expectedLength = contents.LongLength,
            expectedSha256 = Convert.ToHexString(SHA256.HashData(contents)).ToLowerInvariant(),
            sourceUrl = "test-fixture",
            downloadUrl = "",
            license = new
            {
                name = "test owner review required",
                url = "",
                status = "owner-review-required",
                redistributionApproved = false,
            },
        };
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(180_000), $"PowerShell timed out.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}
