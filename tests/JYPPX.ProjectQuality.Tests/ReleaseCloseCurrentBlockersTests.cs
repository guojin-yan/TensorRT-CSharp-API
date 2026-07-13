using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseCurrentBlockersTests
{
    [Fact]
    public void CurrentBlockersSnapshotKeepsReleaseCloseBlockedUntilRealOwnerProofExists()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-close-current-blockers.json");
        JsonElement snapshot = document.RootElement;

        Assert.Equal("release-close-current-blockers", snapshot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-proof-required", snapshot.GetProperty("currentState").GetString());
        Assert.False(snapshot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(snapshot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(snapshot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(snapshot.GetProperty("performsPublish").GetBoolean());
        Assert.False(snapshot.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(snapshot.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(snapshot.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("not runtime proof", snapshot.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", snapshot.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement dashboard = snapshot.GetProperty("sourceSnapshots").GetProperty("finalReleaseCloseBlockerDashboard");
        Assert.Equal("final-release-close-blocker-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-action-required", dashboard.GetProperty("state").GetString());
        Assert.Equal(10, dashboard.GetProperty("blockerCount").GetInt32());
        Assert.Equal(7, dashboard.GetProperty("blockedBlockerCount").GetInt32());
        Assert.Equal(3, dashboard.GetProperty("readyBlockerCount").GetInt32());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement worklist = snapshot.GetProperty("sourceSnapshots").GetProperty("releaseCloseProofWorklist");
        Assert.Equal("release-close-proof-worklist", worklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", worklist.GetProperty("state").GetString());
        Assert.Equal(8, worklist.GetProperty("workItemCount").GetInt32());
        Assert.Equal(8, worklist.GetProperty("blockedWorkItemCount").GetInt32());
        Assert.Equal(0, worklist.GetProperty("readyWorkItemCount").GetInt32());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement validator = snapshot.GetProperty("sourceSnapshots").GetProperty("realProofRecordValidator");
        Assert.Equal("real-proof-record-validator", validator.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-record-validation-input-required", validator.GetProperty("state").GetString());
        Assert.Equal(6, validator.GetProperty("candidateCount").GetInt32());
        Assert.Equal(6, validator.GetProperty("blockedValidatorContractCount").GetInt32());
        Assert.Equal(0, validator.GetProperty("readyValidatorContractCount").GetInt32());
        Assert.False(validator.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    [Fact]
    public void CurrentBlockersSnapshotListsForbiddenSubstitutesAndOwnerActions()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-close-current-blockers.json");
        JsonElement snapshot = document.RootElement;

        string[] forbiddenSubstitutes = snapshot.GetProperty("forbiddenSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains("build-only", forbiddenSubstitutes);
        Assert.Contains("dry-run", forbiddenSubstitutes);
        Assert.Contains("template", forbiddenSubstitutes);
        Assert.Contains("local feed", forbiddenSubstitutes);
        Assert.Contains("ProjectReference", forbiddenSubstitutes);
        Assert.Contains("direct .nupkg", forbiddenSubstitutes);
        Assert.Contains("TensorRtExec report", forbiddenSubstitutes);
        Assert.Contains("YoloVision matrix", forbiddenSubstitutes);
        Assert.Contains("OnnxToEngine report", forbiddenSubstitutes);
        Assert.Contains("readonly diagnostics", forbiddenSubstitutes);
        Assert.Contains("screenshot", forbiddenSubstitutes);
        Assert.Contains("sidecar-only report", forbiddenSubstitutes);
        Assert.Contains("skipped run", forbiddenSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", forbiddenSubstitutes);

        string[] ownerNextActions = snapshot.GetProperty("ownerNextActions")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains(ownerNextActions, static action => action.Contains("real external clean consumer smoke", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(ownerNextActions, static action => action.Contains("manual publish", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(ownerNextActions, static action => action.Contains("post-publish clean consumer proof", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(ownerNextActions, static action => action.Contains("Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady", StringComparison.Ordinal));

        string[] laneIds = snapshot.GetProperty("blockedReleaseCloseLanes")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        Assert.Contains("package-consumer-runtime-proof", laneIds);
        Assert.Contains("public-package-proof-owner-input", laneIds);
        Assert.Contains("post-publish-verification-proof", laneIds);
        Assert.Contains("linux-runner-proof", laneIds);
        Assert.Contains("real-model-runtime-proof", laneIds);
        Assert.Contains("release-issue-close-owner-input", laneIds);
        Assert.Contains("release-issue-close-candidate", laneIds);
        Assert.Contains("release-issue-final-close-decision", laneIds);
        Assert.Contains("strict-close-validator", laneIds);
    }

    [Fact]
    public void CurrentBlockersMarkdownStatesNonProofBoundary()
    {
        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-close-current-blockers.md"));

        Assert.Contains("# Release Close Current Blockers", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-real-owner-proof-required", markdown, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not public package proof", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish verification proof", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady", markdown, StringComparison.Ordinal);
        Assert.Contains("YoloVision matrix", markdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }
}
