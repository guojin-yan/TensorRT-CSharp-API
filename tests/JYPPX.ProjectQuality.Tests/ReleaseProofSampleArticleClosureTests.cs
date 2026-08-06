using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofSampleArticleClosureTests
{
    [Fact]
    public void ClosureMatrixKeepsSampleApplicationAndArticleEvidenceOutOfProofPromotion()
    {
        string matrixPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-sample-article-closure-matrix.json");
        Assert.True(File.Exists(matrixPath), matrixPath);

        string text = File.ReadAllText(matrixPath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("release-proof-sample-article-closure-matrix.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-runtime-proof-required", root.GetProperty("matrixState").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());

        string[] substitutes = root.GetProperty("proofBoundary").GetProperty("forbiddenProofSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string expected in new[]
                 {
                     "build-only",
                     "dry-run",
                     "template",
                     "local feed",
                     "ProjectReference",
                     "direct `.nupkg`",
                     "TensorRtExec report",
                     "YoloVision matrix",
                     "OnnxToEngine report",
                     "readonly diagnostics",
                     "design gate",
                     "blocked-by-cuda-driver",
                 })
        {
            Assert.Contains(expected, substitutes);
        }

        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", text, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", text, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec build/report artifacts cannot replace real-model-runtime", text, StringComparison.Ordinal);
        Assert.Contains("YoloVision matrices cannot replace model-specific output validation", text, StringComparison.Ordinal);
    }

    [Fact]
    public void ClosureMatrixCoversOnnxToEngineTensorRtExecYoloVisionAndArticles()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-sample-article-closure-matrix.json")));

        JsonElement tracks = document.RootElement.GetProperty("tracks");
        AssertTrack(tracks, "release-proof-boundary", "guarded-not-proof");
        AssertTrack(tracks, "onnx-to-engine-trtexec-parity", "tooling-ready-report-not-proof");
        AssertTrack(tracks, "tensorrtexec-application", "application-surface-ready-report-not-proof");
        AssertTrack(tracks, "yolovision-family-sample", "sample-surface-ready-owner-assets-required");
        AssertTrack(tracks, "article-campaign", "planned-and-partially-ready-not-proof");

        string text = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-sample-article-closure-matrix.json"));

        foreach (string expected in new[]
                 {
                     "applications/OnnxToEngine/trtexec-parity-matrix.json",
                     "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
                     "applications/YoloVision/yolo-model-matrix.json",
                     "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
                     "YOLOv26",
                     "custom",
                     "det",
                     "cls",
                     "seg",
                     "obb",
                     "pose",
                     "sem"
                 })
        {
            Assert.Contains(expected, text, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("canPublishPublicly\": true", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue\": true", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseClosureDocumentAndRoadmapStayBlogReadyButNonProof()
    {
        string closureDoc = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-proof-sample-article-closure.md"));
        string roadmapJson = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "article-roadmap-30plus.json"));
        string roadmapMd = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "article-roadmap-30plus.md"));

        foreach (string expected in new[]
                 {
                     "TensorRtExec report",
                     "YoloVision matrix",
                     "OnnxToEngine report",
                     "sample-run-evidence 不能替代 package-consumer-runtime",
                     "package-consumer-runtime 不能替代 post-publish verification",
                     "微信公众号",
                     "博客",
                     "不是 API 文档目录"
                 })
        {
            Assert.Contains(expected, closureDoc + roadmapMd, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument roadmap = JsonDocument.Parse(roadmapJson);
        JsonElement articles = roadmap.RootElement.GetProperty("articles");
        Assert.True(articles.GetArrayLength() >= 30);
        Assert.True(roadmap.RootElement.GetProperty("articleCount").GetInt32() >= 30);
        Assert.False(roadmap.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(roadmap.RootElement.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.Contains(
            "not runtime proof",
            roadmap.RootElement.GetProperty("boundary").GetString(),
            StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloDetLegacyNameDoesNotReturnToLiveSamplesOrClosureEvidence()
    {
        string combined = string.Join(
            "\n",
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "TensorRtSharp.sln")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "YoloVision", "README.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-sample-article-closure-matrix.json")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-proof-sample-article-closure.md")));

        Assert.DoesNotContain("YoloDet", combined, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("YoloVision", combined, StringComparison.Ordinal);
    }

    private static void AssertTrack(JsonElement tracks, string trackId, string expectedStatus)
    {
        foreach (JsonElement track in tracks.EnumerateArray())
        {
            if (track.GetProperty("trackId").GetString() == trackId)
            {
                Assert.Equal(expectedStatus, track.GetProperty("status").GetString());
                Assert.True(track.GetProperty("primaryPaths").GetArrayLength() >= 1, trackId);
                return;
            }
        }

        throw new InvalidOperationException("Track not found: " + trackId);
    }
}
