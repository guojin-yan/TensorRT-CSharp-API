using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseFinalBlockerConvergenceTests
{
    [Fact]
    public void ConvergenceArtifactKeepsReleaseBlocked()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-final-blocker-convergence.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-final-blocker-convergence", root.GetProperty("recordKind").GetString());
        Assert.Equal("artifacts/final-release/release-proof-owner-backfill-summary-validation.json", root.GetProperty("sourceSummaryPath").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("convergenceState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.True(root.GetProperty("remainingBlockerCount").GetInt32() > 0);
        Assert.True(root.GetProperty("missingInputCount").GetInt32() > 0);
    }

    [Fact]
    public void ConvergenceArtifactCarriesAllProofLanesAndOwnerActions()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-final-blocker-convergence.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        foreach (string laneId in new[] { "real-model-runtime", "package-consumer-runtime", "post-publish-verification", "release-issue-close" })
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.Equal("blocked-owner-action-required", lane.GetProperty("currentValidationState").GetString());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("strictValidatorCommand").GetString()));
            Assert.True(lane.GetProperty("missingInputCount").GetInt32() > 0);
            Assert.True(lane.GetProperty("ownerActionRequired").GetProperty("required").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerActionRequired").GetProperty("summary").GetString()));
            Assert.True(lane.GetProperty("promotionCriteria").EnumerateArray().Any());
        }
    }

    [Fact]
    public void ConvergenceArtifactRejectsForbiddenProofSubstitutes()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-final-blocker-convergence.json");
        string[] forbidden = document.RootElement.GetProperty("forbiddenProofSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string expected in new[] { "template", "report", "matrix", "article", "command pack", "summary pack", "dry-run", "build-only", "sidecar-only", "screenshot-only", "Skipped=True", "blocked-by-cuda-driver", "local feed", "ProjectReference", "direct .nupkg" })
        {
            Assert.Contains(expected, forbidden);
        }
    }

    [Fact]
    public void ConvergenceRunnerIsReadOnlyAndNeverPublishes()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseFinalBlockerConvergence.ps1");
        Assert.True(File.Exists(scriptPath));

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("read-only", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ConvergenceArtifactIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("release-final-blocker-convergence.md");
        string article = ReadText("docs", "articles", "zh-cn", "release-final-blocker-convergence.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        Assert.Contains("Release Final Blocker Convergence", artifact, StringComparison.Ordinal);
        Assert.Contains("Can publish publicly: `false`", artifact, StringComparison.Ordinal);
        Assert.Contains("发布阻塞项最终收敛与可发布判定", article, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-final-blocker-convergence.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-final-blocker-convergence.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-final-blocker-convergence.md", docsToc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return ReadText("artifacts", "final-release", fileName);
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
