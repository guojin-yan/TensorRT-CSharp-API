using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofSampleArticleClosureTests
{
    [Fact]
    public void ClosureMatrixKeepsSamplesApplicationsAndArticlesOutOfProofPromotion()
    {
        string matrixPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-sample-article-closure-matrix.json");
        Assert.True(File.Exists(matrixPath), matrixPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(matrixPath));
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
            "OnnxToEngine report"
        })
        {
            Assert.Contains(expected, substitutes);
        }
    }
}
