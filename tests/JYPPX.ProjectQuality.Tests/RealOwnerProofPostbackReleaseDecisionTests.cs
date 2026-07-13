using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealOwnerProofPostbackReleaseDecisionTests
{
    [Fact]
    public void DecisionSnapshotKeepsReleaseBlockedWhenOwnerProofIsMissing()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-owner-proof-postback-release-decision.json");
        JsonElement root = document.RootElement;

        Assert.Equal("real-owner-proof-postback-release-decision", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-proof-not-posted-back", root.GetProperty("decisionState").GetString());
        Assert.Equal("owner-real-proof-missing", root.GetProperty("publishBlockedReason").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.Equal(0, root.GetProperty("ownerProofPostbackDetectedCount").GetInt32());
        Assert.True(root.GetProperty("missingInputCount").GetInt32() > 0);
    }

    [Fact]
    public void DecisionSnapshotCarriesEveryProofLane()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-owner-proof-postback-release-decision.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        foreach (string laneId in new[] { "real-model-runtime", "package-consumer-runtime", "post-publish-verification", "release-issue-close" })
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("ownerProofPostbackDetected").GetBoolean());
            Assert.False(lane.GetProperty("strictValidatorExecuted").GetBoolean());
            Assert.False(lane.GetProperty("strictValidatorPassed").GetBoolean());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("strictValidatorCommand").GetString()));
            Assert.True(lane.GetProperty("missingInputs").EnumerateArray().Any());
        }
    }

    [Fact]
    public void DecisionSnapshotUsesExistingFinalGates()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-owner-proof-postback-release-decision.json");
        JsonElement gates = document.RootElement.GetProperty("upstreamGates");

        Assert.Equal("final-prepublish-readiness-snapshot", gates.GetProperty("finalPrepublishReadinessSnapshot").GetProperty("recordKind").GetString());
        Assert.Equal("owner-real-proof-final-action-worklist", gates.GetProperty("ownerRealProofFinalActionWorklist").GetProperty("recordKind").GetString());
        Assert.Equal("release-final-blocker-convergence", gates.GetProperty("releaseFinalBlockerConvergence").GetProperty("recordKind").GetString());
        Assert.Equal("release-proof-owner-backfill-summary-validation", gates.GetProperty("releaseProofOwnerBackfillSummaryValidation").GetProperty("recordKind").GetString());
    }

    [Fact]
    public void DecisionRunnerAndDocsAreLinked()
    {
        string script = ReadText("eng", "Export-RealOwnerProofPostbackReleaseDecision.ps1");
        string artifact = ReadFinalReleaseText("real-owner-proof-postback-release-decision.md");
        string article = ReadText("docs", "articles", "zh-cn", "real-owner-proof-postback-release-decision.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        Assert.Contains("read-only", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Real Owner Proof Postback Release Decision", artifact, StringComparison.Ordinal);
        Assert.Contains("Can publish publicly: `false`", artifact, StringComparison.Ordinal);
        Assert.Contains("真实 Owner Proof 回填后发布判定", article, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-owner-proof-postback-release-decision.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-owner-proof-postback-release-decision.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-owner-proof-postback-release-decision.md", docsToc, StringComparison.Ordinal);
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
