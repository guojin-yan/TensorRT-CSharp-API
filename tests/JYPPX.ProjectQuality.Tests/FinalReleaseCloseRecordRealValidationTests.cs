using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalReleaseCloseRecordRealValidationTests
{
    private static readonly string[] EvidenceIds =
    [
        "final-release-close-record-real-validator",
        "final-owner-release-close-record-projection",
        "final-release-close-hash-consistency-gate",
        "final-close-owner-approval-boundary-audit",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/final-release-close-record-real-validator.json",
        "artifacts/final-release/final-release-close-record-real-validator.md",
        "artifacts/final-release/final-release-close-record-real-validator-validation.json",
        "artifacts/final-release/final-release-close-record-real-validator-validation.md",
        "artifacts/final-release/final-owner-release-close-record-projection.json",
        "artifacts/final-release/final-owner-release-close-record-projection.md",
        "artifacts/final-release/final-owner-release-close-record-projection-validation.json",
        "artifacts/final-release/final-owner-release-close-record-projection-validation.md",
        "artifacts/final-release/final-release-close-hash-consistency-gate.json",
        "artifacts/final-release/final-release-close-hash-consistency-gate.md",
        "artifacts/final-release/final-release-close-hash-consistency-gate-validation.json",
        "artifacts/final-release/final-release-close-hash-consistency-gate-validation.md",
        "artifacts/final-release/final-close-owner-approval-boundary-audit.json",
        "artifacts/final-release/final-close-owner-approval-boundary-audit.md",
        "artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json",
        "artifacts/final-release/final-close-owner-approval-boundary-audit-validation.md",
    ];

    [Fact]
    public void FinalReleaseCloseRecordRealValidationItemsStayBlockedNonProof()
    {
        RunPipeline();

        using JsonDocument validatorDocument = ReadFinalReleaseJson("final-release-close-record-real-validator-validation.json");
        JsonElement validator = validatorDocument.RootElement;
        Assert.Equal("blocked-final-release-close-record-real-proof-required", validator.GetProperty("validationState").GetString());
        Assert.Equal(13, validator.GetProperty("requiredFieldCount").GetInt32());
        Assert.Equal(13, validator.GetProperty("blockedRequiredFieldCount").GetInt32());
        Assert.Equal(0, validator.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, validator.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(validator);

        using JsonDocument projectionDocument = ReadFinalReleaseJson("final-owner-release-close-record-projection-validation.json");
        JsonElement projection = projectionDocument.RootElement;
        Assert.Equal("blocked-final-owner-release-close-record-projection-owner-input-required", projection.GetProperty("validationState").GetString());
        Assert.Equal(4, projection.GetProperty("laneCount").GetInt32());
        Assert.Equal(4, projection.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, projection.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, projection.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(projection);

        using JsonDocument hashGateDocument = ReadFinalReleaseJson("final-release-close-hash-consistency-gate-validation.json");
        JsonElement hashGate = hashGateDocument.RootElement;
        Assert.Equal("blocked-final-release-close-hash-consistency-owner-proof-required", hashGate.GetProperty("validationState").GetString());
        Assert.Equal(8, hashGate.GetProperty("hashLaneCount").GetInt32());
        Assert.Equal(0, hashGate.GetProperty("mismatchedHashCount").GetInt32());
        Assert.Equal(7, hashGate.GetProperty("blockedHashLaneCount").GetInt32());
        Assert.Equal(0, hashGate.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, hashGate.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(hashGate);

        using JsonDocument approvalAuditDocument = ReadFinalReleaseJson("final-close-owner-approval-boundary-audit-validation.json");
        JsonElement approvalAudit = approvalAuditDocument.RootElement;
        Assert.Equal("blocked-final-close-owner-approval-boundary-owner-action-required", approvalAudit.GetProperty("validationState").GetString());
        Assert.Equal(6, approvalAudit.GetProperty("approvalLaneCount").GetInt32());
        Assert.Equal(6, approvalAudit.GetProperty("blockedApprovalLaneCount").GetInt32());
        Assert.Equal(0, approvalAudit.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(1, approvalAudit.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(approvalAudit);
    }

    [Fact]
    public void ReleaseEvidenceBundleAuditAndDocsIncludeFinalReleaseCloseRecordRealValidation()
    {
        RunPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-final-release-close-record-real-proof-required", evidence.GetProperty("finalReleaseCloseRecordRealValidatorValidationState").GetString());
        Assert.Equal("blocked-final-owner-release-close-record-projection-owner-input-required", evidence.GetProperty("finalOwnerReleaseCloseRecordProjectionValidationState").GetString());
        Assert.Equal("blocked-final-release-close-hash-consistency-owner-proof-required", evidence.GetProperty("finalReleaseCloseHashConsistencyGateValidationState").GetString());
        Assert.Equal("blocked-final-close-owner-approval-boundary-owner-action-required", evidence.GetProperty("finalCloseOwnerApprovalBoundaryAuditValidationState").GetString());

        Assert.Equal(13, evidence.GetProperty("finalReleaseCloseRecordRealValidatorRequiredFieldCount").GetInt32());
        Assert.Equal(13, evidence.GetProperty("finalReleaseCloseRecordRealValidatorBlockedRequiredFieldCount").GetInt32());
        Assert.Equal(4, evidence.GetProperty("finalOwnerReleaseCloseRecordProjectionLaneCount").GetInt32());
        Assert.Equal(4, evidence.GetProperty("finalOwnerReleaseCloseRecordProjectionBlockedLaneCount").GetInt32());
        Assert.Equal(8, evidence.GetProperty("finalReleaseCloseHashConsistencyGateHashLaneCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("finalReleaseCloseHashConsistencyGateMismatchedHashCount").GetInt32());
        Assert.Equal(7, evidence.GetProperty("finalReleaseCloseHashConsistencyGateBlockedHashLaneCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("finalCloseOwnerApprovalBoundaryAuditLaneCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("finalCloseOwnerApprovalBoundaryAuditBlockedLaneCount").GetInt32());

        Assert.False(evidence.GetProperty("finalReleaseCloseRecordRealValidatorCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerReleaseCloseRecordProjectionCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("finalReleaseCloseHashConsistencyGateCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalCloseOwnerApprovalBoundaryAuditCanCloseReleaseIssue").GetBoolean());

        foreach (string id in EvidenceIds)
        {
            AssertBlockedEvidenceItem(evidence, id);
        }

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string sourceArtifact in SourceArtifacts)
        {
            Assert.Contains(sourceArtifact, sourceArtifacts);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string id in EvidenceIds)
        {
            AssertAuditedNonProofItem(audit, id);
        }

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        foreach (string id in EvidenceIds)
        {
            Assert.Contains($"articles/zh-cn/{id}.md", docsIndex, StringComparison.Ordinal);
            Assert.Contains($"articles/zh-cn/{id}.md", docsToc, StringComparison.Ordinal);
            Assert.Contains(id, readme, StringComparison.Ordinal);
            Assert.Contains(id, readmeZh, StringComparison.Ordinal);
            Assert.Contains(id, releaseEvidenceDoc, StringComparison.Ordinal);
        }
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-PublicPublishRealResultRecordDraft.ps1");
        RunPowerShell("Test-PublicPublishRealResultRecordDraft.ps1", "-Strict");
        RunPowerShell("Export-PostPublishCleanConsumerProofRecordDraft.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofRecordDraft.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishForbiddenSubstituteScan.ps1");
        RunPowerShell("Test-PublicPublishForbiddenSubstituteScan.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseRealProofImportBridge.ps1");
        RunPowerShell("Test-ReleaseCloseRealProofImportBridge.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerCloseReadinessCheckpoint.ps1");
        RunPowerShell("Test-FinalOwnerCloseReadinessCheckpoint.ps1", "-Strict");
        RunPowerShell("Export-FinalReleaseCloseRecordRealValidator.ps1");
        RunPowerShell("Test-FinalReleaseCloseRecordRealValidator.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerReleaseCloseRecordProjection.ps1");
        RunPowerShell("Test-FinalOwnerReleaseCloseRecordProjection.ps1", "-Strict");
        RunPowerShell("Export-FinalReleaseCloseHashConsistencyGate.ps1");
        RunPowerShell("Test-FinalReleaseCloseHashConsistencyGate.ps1", "-Strict");
        RunPowerShell("Export-FinalCloseOwnerApprovalBoundaryAudit.ps1");
        RunPowerShell("Test-FinalCloseOwnerApprovalBoundaryAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseRecordProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        string boundary = item.GetProperty("boundary").GetString()!;
        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
