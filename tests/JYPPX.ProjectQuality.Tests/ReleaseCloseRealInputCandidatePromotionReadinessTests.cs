using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseRealInputCandidatePromotionReadinessTests
{
    [Fact]
    public void CandidatePromotionReadinessStaysBlockedUntilRealOwnerInputsPassStrictValidators()
    {
        RunPipeline();

        using JsonDocument document = ReadFinalReleaseJson("release-close-real-input-candidate-promotion-readiness.json");
        JsonElement readiness = document.RootElement;

        Assert.Equal("release-close-real-input-candidate-promotion-readiness", readiness.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-input-candidate-promotion-real-owner-input-required", readiness.GetProperty("readinessState").GetString());
        Assert.True(readiness.GetProperty("sourceRecordCount").GetInt32() >= 11);
        Assert.Equal(11, readiness.GetProperty("candidatePromotionLaneCount").GetInt32());
        Assert.Equal(11, readiness.GetProperty("blockedCandidatePromotionLaneCount").GetInt32());
        Assert.Equal(0, readiness.GetProperty("readyCandidatePromotionLaneCount").GetInt32());
        Assert.Equal(0, readiness.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(readiness.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(readiness);
        AssertBoundary(readiness.GetProperty("boundary").GetString()!, requireNotReady: true);

        string[] laneIds = readiness.GetProperty("candidatePromotionMatrix")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();
        foreach (string laneId in RequiredLaneIds)
        {
            Assert.Contains(laneId, laneIds);
        }

        foreach (JsonElement lane in readiness.GetProperty("candidatePromotionMatrix").EnumerateArray())
        {
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.True(lane.GetProperty("requiredRealInputCount").GetInt32() > 0);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("strictValidator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("promotionBlockedUntil").GetString()));
            Assert.Contains("real Owner evidence", lane.GetProperty("nonSubstituteBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
            AssertFlagsStayNonProof(lane);
            AssertBoundary(lane.GetProperty("boundary").GetString()!, requireNotReady: false);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-real-input-candidate-promotion-readiness-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("release-close-real-input-candidate-promotion-readiness-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-input-candidate-promotion-real-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("sourceRecordCount").GetInt32() >= 11);
        Assert.Equal(11, validation.GetProperty("candidatePromotionLaneCount").GetInt32());
        Assert.Equal(11, validation.GetProperty("blockedCandidatePromotionLaneCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyCandidatePromotionLaneCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!, requireNotReady: true);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryCandidatePromotionReadinessAsNonProof()
    {
        RunPipeline();
        RunPowerShell("Test-FinalPublishProofGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-release-close-real-input-candidate-promotion-real-owner-input-required", evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessState").GetString());
        Assert.Equal("blocked-release-close-real-input-candidate-promotion-real-owner-input-required", evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessValidationState").GetString());
        Assert.True(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessSourceRecordCount").GetInt32() >= 11);
        Assert.Equal(11, evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessLaneCount").GetInt32());
        Assert.Equal(11, evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessBlockedLaneCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessReadyLaneCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessFailedActionRequiredCount").GetInt32() > 0);
        Assert.True(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessNotExecutedByAutomation").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessPerformsPublish").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessIsPostPublishProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealInputCandidatePromotionReadinessIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-close-real-input-candidate-promotion-readiness");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("lanes=11", evidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!, requireNotReady: true);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-close-real-input-candidate-promotion-readiness.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-real-input-candidate-promotion-readiness.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-real-input-candidate-promotion-readiness-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-real-input-candidate-promotion-readiness-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "release-close-real-input-candidate-promotion-readiness" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());

        using JsonDocument finalPublishGateDocument = ReadFinalReleaseJson("final-publish-proof-gate-report.json");
        JsonElement finalPublishGate = finalPublishGateDocument.RootElement;
        Assert.Equal("blocked-final-publish-real-proof-required", finalPublishGate.GetProperty("validationState").GetString());
        Assert.Equal(0, finalPublishGate.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(finalPublishGate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(finalPublishGate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(finalPublishGate.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static readonly string[] RequiredLaneIds =
    [
        "public-package-proof",
        "clean-external-consumer-runtime-proof",
        "post-publish-clean-consumer-proof",
        "hash-path-validation",
        "forbidden-substitute-validation",
        "strict-close-dry-run",
        "rollback-review",
        "final-close-decision",
        "release-close-strict-record-candidate",
        "final-release-close-record-real-validator",
        "final-publish-proof-gate"
    ];

    private static void RunPipeline()
    {
        RunPowerShell("Export-RealOwnerEvidenceStrictValidatorOrchestration.ps1");
        RunPowerShell("Test-RealOwnerEvidenceStrictValidatorOrchestration.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseRealInputCandidatePromotionReadiness.ps1");
        RunPowerShell("Test-ReleaseCloseRealInputCandidatePromotionReadiness.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertBoundary(string boundary, bool requireNotReady)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        if (requireNotReady)
        {
            Assert.Contains("not ready", boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
