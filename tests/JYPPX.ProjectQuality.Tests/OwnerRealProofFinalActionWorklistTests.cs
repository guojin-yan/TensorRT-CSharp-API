using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofFinalActionWorklistTests
{
    [Fact]
    public void WorklistKeepsReleaseBlockedAndOwnerActionOnly()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-final-action-worklist.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-real-proof-final-action-worklist", root.GetProperty("recordKind").GetString());
        Assert.Equal("artifacts/final-release/release-final-blocker-convergence.json", root.GetProperty("sourceConvergencePath").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("worklistState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.True(root.GetProperty("remainingWorkItemCount").GetInt32() > 0);
        Assert.True(root.GetProperty("missingInputCount").GetInt32() > 0);

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("read-only", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner-action-only", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not create proof", boundary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WorklistCarriesEveryReleaseProofLane()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-final-action-worklist.json");
        JsonElement[] workItems = document.RootElement.GetProperty("workItems").EnumerateArray().ToArray();
        Assert.Equal(4, workItems.Length);

        foreach (string laneId in new[] { "real-model-runtime", "package-consumer-runtime", "post-publish-verification", "release-issue-close" })
        {
            JsonElement item = Assert.Single(workItems, element => element.GetProperty("id").GetString() == laneId);
            Assert.False(item.GetProperty("canPromote").GetBoolean());
            Assert.True(item.GetProperty("missingInputCount").GetInt32() > 0);
            Assert.True(item.GetProperty("ownerCommands").EnumerateArray().Any());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("strictValidatorCommand").GetString()));
            Assert.True(item.GetProperty("promotionCriteria").EnumerateArray().Any());
            Assert.True(item.GetProperty("ownerActionRequired").GetProperty("required").GetBoolean());
        }
    }

    [Fact]
    public void WorklistRejectsForbiddenProofSubstitutes()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-final-action-worklist.json");
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
    public void WorklistRunnerNeverPublishesOrPromotes()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofFinalActionWorklist.ps1");
        Assert.True(File.Exists(scriptPath));

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("read-only", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("owner-action-only", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WorklistIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("owner-real-proof-final-action-worklist.md");
        string article = ReadText("docs", "articles", "zh-cn", "owner-real-proof-final-action-worklist.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        Assert.Contains("Owner Real Proof Final Action Worklist", artifact, StringComparison.Ordinal);
        Assert.Contains("Can publish publicly: `false`", artifact, StringComparison.Ordinal);
        Assert.Contains("Owner 真实 Proof 最终执行清单", article, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-real-proof-final-action-worklist.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-real-proof-final-action-worklist.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-real-proof-final-action-worklist.md", docsToc, StringComparison.Ordinal);
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
