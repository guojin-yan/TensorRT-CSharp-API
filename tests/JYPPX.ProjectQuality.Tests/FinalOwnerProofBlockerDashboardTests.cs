using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class FinalOwnerProofBlockerDashboardTests
{
    [Fact]
    public void FinalOwnerProofBlockerDashboardConsolidatesBlockedOwnerLanes()
    {
        using JsonDocument document = ReleaseCandidateFreezeManifestTests.ReadFinalReleaseJson("final-owner-proof-blocker-dashboard.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-owner-proof-blocker-dashboard.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("dashboardState").GetString());
        ReleaseCandidateFreezeManifestTests.AssertFalseProofPublishCloseFlags(root);
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Equal(4, root.GetProperty("blockedLaneCount").GetInt32());

        string[] laneIds = root.GetProperty("blockedLanes")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        foreach (string expected in new[] { "sample-run-evidence", "package-consumer-runtime", "post-publish-verification", "release-close-owner-approval" })
        {
            Assert.Contains(expected, laneIds);
        }

        foreach (JsonElement lane in root.GetProperty("blockedLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("isProof").GetBoolean());
            Assert.Contains("pwsh", lane.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("owner", lane.GetProperty("ownerAction").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Empty(lane.GetProperty("canBeReplacedBy").EnumerateArray());
        }

        string raw = root.GetRawText();
        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", raw, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", raw, StringComparison.Ordinal);
        ReleaseCandidateFreezeManifestTests.AssertForbiddenSubstitutes(root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray());
    }

    [Fact]
    public void FinalOwnerProofBlockerDashboardArticleIsLinkedAndNonProof()
    {
        ReleaseCandidateFreezeManifestTests.AssertArticleLinked("final-owner-proof-blocker-dashboard.md");
    }
}
