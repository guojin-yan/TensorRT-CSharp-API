using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealOwnerEvidenceStrictValidatorOrchestrationTests
{
    [Fact]
    public void OrchestrationMapsRealOwnerFieldsWithoutPromotingProof()
    {
        RunPipeline();

        using JsonDocument document = ReadFinalReleaseJson("real-owner-evidence-strict-validator-orchestration.json");
        JsonElement orchestration = document.RootElement;

        Assert.Equal("real-owner-evidence-strict-validator-orchestration", orchestration.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-evidence-strict-validator-real-owner-input-required", orchestration.GetProperty("orchestrationState").GetString());
        Assert.True(orchestration.GetProperty("sourceRecordCount").GetInt32() >= 11);
        Assert.True(orchestration.GetProperty("fieldReadinessCount").GetInt32() >= 16);
        Assert.Equal(orchestration.GetProperty("fieldReadinessCount").GetInt32(), orchestration.GetProperty("blockedFieldReadinessCount").GetInt32());
        Assert.Equal(0, orchestration.GetProperty("readyFieldReadinessCount").GetInt32());
        Assert.True(orchestration.GetProperty("validatorConsumerCount").GetInt32() >= 6);
        Assert.True(orchestration.GetProperty("ownerInputSurfaceCount").GetInt32() >= 5);
        Assert.Equal(0, orchestration.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(orchestration.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(orchestration);
        AssertBoundary(orchestration.GetProperty("boundary").GetString()!, requireNotProofReady: true);

        string[] fieldNames = orchestration.GetProperty("fieldReadinessMatrix")
            .EnumerateArray()
            .Select(static field => field.GetProperty("fieldName").GetString()!)
            .ToArray();
        foreach (string fieldName in RequiredFields)
        {
            Assert.Contains(fieldName, fieldNames);
        }

        foreach (JsonElement field in orchestration.GetProperty("fieldReadinessMatrix").EnumerateArray())
        {
            Assert.Equal("blocked-real-owner-input-required", field.GetProperty("currentState").GetString());
            Assert.False(field.GetProperty("canBeSatisfiedByLocalDryRun").GetBoolean());
            Assert.False(field.GetProperty("canBeSatisfiedByProjectReference").GetBoolean());
            Assert.False(field.GetProperty("canBeSatisfiedByDirectNupkg").GetBoolean());
            Assert.False(field.GetProperty("canBeSatisfiedByLocalFeed").GetBoolean());
            Assert.Contains("real Owner evidence", field.GetProperty("nonSubstituteBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
            AssertFlagsStayNonProof(field);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-owner-evidence-strict-validator-orchestration-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-owner-evidence-strict-validator-orchestration-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-evidence-strict-validator-real-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("sourceRecordCount").GetInt32() >= 11);
        Assert.True(validation.GetProperty("fieldReadinessCount").GetInt32() >= 16);
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!, requireNotProofReady: true);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryOrchestrationAsNonProof()
    {
        RunPipeline();
        RunPowerShell("Test-FinalPublishProofGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-owner-evidence-strict-validator-real-owner-input-required", evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationState").GetString());
        Assert.Equal("blocked-real-owner-evidence-strict-validator-real-owner-input-required", evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationValidationState").GetString());
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationSourceRecordCount").GetInt32() >= 11);
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationFieldReadinessCount").GetInt32() >= 16);
        Assert.Equal(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationFieldReadinessCount").GetInt32(), evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationBlockedFieldReadinessCount").GetInt32());
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationValidatorConsumerCount").GetInt32() >= 6);
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationOwnerInputSurfaceCount").GetInt32() >= 5);
        Assert.Equal(0, evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationFailedActionRequiredCount").GetInt32() > 0);
        Assert.True(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationNotExecutedByAutomation").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationPerformsPublish").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationIsPostPublishProof").GetBoolean());
        Assert.False(evidence.GetProperty("realOwnerEvidenceStrictValidatorOrchestrationIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-owner-evidence-strict-validator-orchestration");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("fields=", evidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!, requireNotProofReady: true);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/real-owner-evidence-strict-validator-orchestration.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-owner-evidence-strict-validator-orchestration.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "real-owner-evidence-strict-validator-orchestration" &&
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

    private static readonly string[] RequiredFields =
    [
        "publicPackageSourceUrl",
        "downloadedNupkgSha256",
        "packageId",
        "packageVersion",
        "packageSourceKind",
        "cleanConsumerProjectPath",
        "cleanConsumerLogPath",
        "cleanConsumerLogSha256",
        "postPublishInstallLogPath",
        "postPublishRunLogPath",
        "stdoutPath",
        "stderrPath",
        "hostMetadata",
        "nonSubstituteConfirmations",
        "rollbackReview",
        "finalCloseDecision"
    ];

    private static void RunPipeline()
    {
        RunPowerShell("Export-OwnerInputContractConvergence.ps1");
        RunPowerShell("Test-OwnerInputContractConvergence.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerStrictCloseExecutionOrder.ps1");
        RunPowerShell("Test-FinalOwnerStrictCloseExecutionOrder.ps1", "-Strict");
        RunPowerShell("Export-RealOwnerEvidenceStrictValidatorOrchestration.ps1");
        RunPowerShell("Test-RealOwnerEvidenceStrictValidatorOrchestration.ps1", "-Strict");
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

    private static void AssertBoundary(string boundary, bool requireNotProofReady)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        if (requireNotProofReady)
        {
            Assert.Contains("not proof ready", boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
