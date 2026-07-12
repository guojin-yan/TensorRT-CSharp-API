using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RemoteCiAndPublicPublishProofBackfillGateTests
{
    [Fact]
    public void RemoteCiAndPublicPublishGateStaysBlockedNonProofAndReachesEvidenceBundle()
    {
        RunPipeline();

        using JsonDocument gateDocument = ReadFinalReleaseJson("remote-ci-and-public-publish-proof-backfill-gate.json");
        JsonElement gate = gateDocument.RootElement;
        Assert.Equal("remote-ci-and-public-publish-proof-backfill-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", gate.GetProperty("gateState").GetString());
        Assert.Equal(6, gate.GetProperty("laneCount").GetInt32());
        Assert.True(gate.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, gate.GetProperty("boundaryFailureCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(gate.GetProperty("readyForOwnerReview").GetBoolean());
        AssertFlagsStayNonProof(gate);
        AssertContainsRequiredLanes(gate);

        using JsonDocument validationDocument = ReadFinalReleaseJson("remote-ci-and-public-publish-proof-backfill-gate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("remote-ci-and-public-publish-proof-backfill-gate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("laneCount").GetInt32());
        Assert.True(validation.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, validation.GetProperty("boundaryFailureCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        AssertFlagsStayNonProof(validation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateLaneCount").GetInt32());
        Assert.True(evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateBlockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateBoundaryFailureCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateFailedBlockerCount").GetInt32());
        Assert.False(evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("remoteCiAndPublicPublishProofBackfillGateIsGitHubActionsProof").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "remote-ci-and-public-publish-proof-backfill-gate", "not GitHub Actions proof");
        AssertRequiredSourceArtifacts(evidence);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "remote-ci-and-public-publish-proof-backfill-gate");
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-GitHubPublishAndCiStatusSnapshot.ps1");
        RunPowerShell("Test-GitHubPublishAndCiStatusSnapshot.ps1", "-Strict");
        RunPowerShell("Export-FinalPrepublishQualityFreezeDashboard.ps1");
        RunPowerShell("Test-FinalPrepublishQualityFreezeDashboard.ps1", "-Strict");
        RunPowerShell("Export-RemoteCiAndPublicPublishProofBackfillGate.ps1");
        RunPowerShell("Test-RemoteCiAndPublicPublishProofBackfillGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static void AssertContainsRequiredLanes(JsonElement gate)
    {
        string[] laneIds = gate.GetProperty("lanes")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string laneId in RequiredLaneIds)
        {
            Assert.Contains(laneId, laneIds);
        }
    }

    private static void AssertRequiredSourceArtifacts(JsonElement evidence)
    {
        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string artifact in RequiredSourceArtifacts)
        {
            Assert.Contains(artifact, sourceArtifacts);
        }
    }

    private static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(element.GetProperty("isGitHubActionsProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id, string boundaryText)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryText, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }

    private static readonly string[] RequiredLaneIds =
    [
        "github-source-head-status",
        "github-actions-run-proof",
        "owner-public-publish-result",
        "public-package-download-proof",
        "post-publish-clean-consumer-proof",
        "final-prepublish-freeze",
    ];

    private static readonly string[] RequiredSourceArtifacts =
    [
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
    ];
}
