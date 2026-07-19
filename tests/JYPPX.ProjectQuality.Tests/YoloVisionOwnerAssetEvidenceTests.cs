using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionOwnerAssetEvidenceTests
{
    [Fact]
    public void YoloVisionOwnerAssetTemplateCapturesRequiredFieldsWithoutPromotingProof()
    {
        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "yolovision-owner-asset-evidence.template.json");
        Assert.True(File.Exists(templatePath), templatePath);

        string text = File.ReadAllText(templatePath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-owner-asset-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
        Assert.Equal("template-only-not-proof", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] families = root.GetProperty("allowedFamilies").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "yolox", "custom" }, families);

        string[] tasks = root.GetProperty("allowedTasks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(new[] { "det", "cls", "seg", "obb", "pose", "sem" }, tasks);

        JsonElement ownerFields = root.GetProperty("ownerFields");
        foreach (string requiredField in new[]
        {
            "modelFamily",
            "task",
            "modelSourceUrl",
            "modelLicense",
            "modelSha256",
            "labelsSource",
            "labelsSha256",
            "inputImageSource",
            "inputImageLicense",
            "inputImageSha256",
            "generatedEnginePath",
            "generatedEngineSha256",
            "commandLine",
            "stdoutLogPath",
            "stderrLogPath",
            "outputJsonPath",
            "outputJsonSha256",
            "ownerReviewStatus"
        })
        {
            Assert.True(ownerFields.TryGetProperty(requiredField, out JsonElement value), requiredField);
            Assert.Contains("owner-required", value.GetString(), StringComparison.Ordinal);
        }

        Assert.Contains("YoloVision Passed=True", text, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", text, StringComparison.Ordinal);
        Assert.Contains("YoloVision matrix", text, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", text, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", text, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionOwnerAssetGuideExplainsTemplateOnlyBoundary()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-owner-asset-evidence-guide.md"));

        Assert.Contains("适用读者", article, StringComparison.Ordinal);
        Assert.Contains("解决问题", article, StringComparison.Ordinal);
        Assert.Contains("边界说明", article, StringComparison.Ordinal);
        Assert.Contains("下一步", article, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", article, StringComparison.Ordinal);
        Assert.Contains("SHA256", article, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
        Assert.Contains("post-publish verification", article, StringComparison.Ordinal);
        Assert.Contains("template", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("不是 runtime proof", article, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }
}
