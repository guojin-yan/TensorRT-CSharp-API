using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublishingRoadmapDocsTests
{
    [Fact]
    public void PublishingRoadmapKeepsThirtyPlusArticlePlanDiscoverableAndNonProof()
    {
        string roadmapJsonPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.json");
        string roadmapMarkdownPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.md");
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        using JsonDocument roadmapDocument = JsonDocument.Parse(File.ReadAllText(roadmapJsonPath));
        JsonElement roadmap = roadmapDocument.RootElement;

        Assert.True(roadmap.GetProperty("articleCount").GetInt32() >= 30);
        Assert.False(roadmap.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(roadmap.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("not runtime proof", roadmap.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("articles/zh-cn/publishing/article-roadmap-30plus.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/publishing/article-roadmap-30plus.md", docsToc, StringComparison.Ordinal);

        string roadmapMarkdown = File.ReadAllText(roadmapMarkdownPath);
        Assert.Contains("articleCount", roadmapMarkdown, StringComparison.Ordinal);
        Assert.Contains("40", roadmapMarkdown, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", roadmapMarkdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("near-ready-owner-proof-input", roadmapMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner proof input", roadmapMarkdown, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", roadmapMarkdown, StringComparison.OrdinalIgnoreCase);

        JsonElement[] articles = roadmap.GetProperty("articles").EnumerateArray().ToArray();
        foreach (int id in new[] { 16, 17, 36, 40 })
        {
            JsonElement article = Assert.Single(articles, item => item.GetProperty("id").GetInt32() == id);
            Assert.Equal("near-ready-owner-proof-input", article.GetProperty("status").GetString());
            Assert.Contains("owner proof input", article.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void SourceBuildAndDualPackageGuidesCoverNativeBridgeAndReleaseBoundaries()
    {
        string sourceBuild = ReadDoc("source-build-windows-cpp-bridge.md");
        string presets = ReadDoc("source-build-cmake-presets-and-bindings.md");
        string packages = ReadDoc("nuget-github-dual-package-strategy.md");
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        foreach (string href in new[]
        {
            "articles/zh-cn/source-build-windows-cpp-bridge.md",
            "articles/zh-cn/source-build-cmake-presets-and-bindings.md",
            "articles/zh-cn/nuget-github-dual-package-strategy.md"
        })
        {
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
        }

        Assert.Contains("cmake --preset", sourceBuild, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Generate-Bindings.ps1", sourceBuild, StringComparison.Ordinal);
        Assert.Contains("JYPPX_ENABLE_DEVELOPMENT_PROBING", sourceBuild, StringComparison.Ordinal);
        Assert.Contains("TRT8", sourceBuild, StringComparison.Ordinal);
        Assert.Contains("TRT10", sourceBuild, StringComparison.Ordinal);
        Assert.Contains("TRT11", sourceBuild, StringComparison.Ordinal);

        Assert.Contains("manifest", presets, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("native source", presets, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("高层 wrapper", presets, StringComparison.Ordinal);
        Assert.Contains("不删除 deferred", presets, StringComparison.Ordinal);

        Assert.Contains("GitHub Release assets", packages, StringComparison.Ordinal);
        Assert.Contains("NuGet-compatible source", packages, StringComparison.Ordinal);
        Assert.Contains("managed + bridge-only", packages, StringComparison.Ordinal);
        Assert.Contains("CUDA、TensorRT、cuDNN、NVRTC", packages, StringComparison.Ordinal);
        Assert.Contains("不能重新 pack、push 或上传", packages, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionOnnxToEngineAndTensorRtExecRoadmapsReflectChangedGoal()
    {
        string yolo = ReadDoc("yolovision-series-roadmap.md");
        string onnx = ReadDoc("onnx-to-engine-trtexec-parity-roadmap.md");
        string app = ReadDoc("tensorrtexec-console-winforms-application-roadmap.md");
        string samplesReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));
        string onnxReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "README.md"));
        string applicationsReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "README.md"));

        foreach (string marker in new[]
        {
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLOv11",
            "YOLOv26",
            "det",
            "cls",
            "seg",
            "obb",
            "pose",
            "sem"
        })
        {
            Assert.Contains(marker, yolo, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("旧 detection-only 样例目录已经被 YoloVision 取代", yolo, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", yolo, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", samplesReadme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("applications\\YoloVision", yolo, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", yolo, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", onnx, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("parse-only", onnx, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("implementationClass", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "trtexec-parity-matrix.json")), StringComparison.Ordinal);
        Assert.Contains("ownerEvidenceRequiredForPromotion", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "tensor-rt-exec-gui-cli-field-map.json")), StringComparison.Ordinal);
        Assert.Contains("trtexec", onnx + onnxReadme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Console", app, StringComparison.Ordinal);
        Assert.Contains("WinForms", app + applicationsReadme, StringComparison.Ordinal);
        Assert.Contains("applications/TensorRtExec", app, StringComparison.Ordinal);
        Assert.Contains("not package-consumer-runtime proof", applicationsReadme, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadDoc(string fileName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", fileName);
        Assert.True(File.Exists(path), path);
        return File.ReadAllText(path);
    }
}
