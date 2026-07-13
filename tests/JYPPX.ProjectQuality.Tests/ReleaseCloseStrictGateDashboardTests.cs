using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCloseStrictGateDashboardTests
{
    [Fact]
    public void StrictGateDashboardKeepsAllCloseLanesBlocked()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-close-strict-gate-dashboard.json")));

        JsonElement root = document.RootElement;
        Assert.Equal("release-close-strict-gate-dashboard.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", root.GetProperty("dashboardState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] laneIds = root.GetProperty("blockedLanes").EnumerateArray().Select(static lane => lane.GetProperty("laneId").GetString()!).ToArray();
        foreach (string expected in new[] { "release-evidence-bundle", "release-close-preflight", "release-issue-close-record", "package-consumer-runtime", "post-publish-verification", "sample-run-evidence", "owner-approval" })
        {
            Assert.Contains(expected, laneIds);
        }

        foreach (JsonElement lane in root.GetProperty("blockedLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains("pwsh", lane.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("blockedReason").GetString()));
        }

        Assert.Contains("No template-only record", root.GetProperty("strictCloseRule").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void StrictGateArticleIsLinkedAndNonProof()
    {
        OwnerRealProofFieldDeltaDashboardTests.AssertArticleLinked("release-close-strict-gate-dashboard.md");
    }
}
