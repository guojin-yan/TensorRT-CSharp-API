using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPublishRealProofBackfillExecutionTests
{
    private static readonly string[] EvidenceIds =
    [
        "public-publish-real-result-record-draft",
        "post-publish-clean-consumer-proof-record-draft",
        "public-publish-forbidden-substitute-scan",
        "release-close-real-proof-import-bridge",
        "final-owner-close-readiness-checkpoint",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/public-publish-real-result-record-draft.json",
        "artifacts/final-release/public-publish-real-result-record-draft.md",
        "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
        "artifacts/final-release/public-publish-real-result-record-draft-validation.md",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-draft.md",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.md",
        "artifacts/final-release/public-publish-forbidden-substitute-scan.json",
        "artifacts/final-release/public-publish-forbidden-substitute-scan.md",
        "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json",
        "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.md",
        "artifacts/final-release/release-close-real-proof-import-bridge.json",
        "artifacts/final-release/release-close-real-proof-import-bridge.md",
        "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
        "artifacts/final-release/release-close-real-proof-import-bridge-validation.md",
        "artifacts/final-release/final-owner-close-readiness-checkpoint.json",
        "artifacts/final-release/final-owner-close-readiness-checkpoint.md",
        "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json",
        "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.md",
    ];

    [Fact]
    public void PublicPublishRealProofBackfillExecutionItemsStayBlockedNonProof()
    {
        RunPipeline();

        using JsonDocument publishDraftDocument = ReadFinalReleaseJson("public-publish-real-result-record-draft-validation.json");
        JsonElement publishDraft = publishDraftDocument.RootElement;
        Assert.Equal("blocked-public-publish-real-result-record-required", publishDraft.GetProperty("validationState").GetString());
        Assert.Equal(12, publishDraft.GetProperty("requiredFieldCount").GetInt32());
        Assert.Equal(12, publishDraft.GetProperty("blockedRequiredFieldCount").GetInt32());
        Assert.Equal(0, publishDraft.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(publishDraft.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(publishDraft);

        using JsonDocument consumerDraftDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-record-draft-validation.json");
        JsonElement consumerDraft = consumerDraftDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", consumerDraft.GetProperty("validationState").GetString());
        Assert.Equal(19, consumerDraft.GetProperty("requiredFieldCount").GetInt32());
        Assert.Equal(19, consumerDraft.GetProperty("blockedRequiredFieldCount").GetInt32());
        Assert.Equal(0, consumerDraft.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(consumerDraft.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(consumerDraft);

        using JsonDocument substituteScanDocument = ReadFinalReleaseJson("public-publish-forbidden-substitute-scan-validation.json");
        JsonElement substituteScan = substituteScanDocument.RootElement;
        Assert.Equal("blocked-public-publish-forbidden-substitute-scan-owner-proof-required", substituteScan.GetProperty("validationState").GetString());
        Assert.Equal(9, substituteScan.GetProperty("substituteCheckCount").GetInt32());
        Assert.Equal(9, substituteScan.GetProperty("blockedSubstituteCheckCount").GetInt32());
        Assert.Equal(0, substituteScan.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(substituteScan.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(substituteScan);

        using JsonDocument bridgeDocument = ReadFinalReleaseJson("release-close-real-proof-import-bridge-validation.json");
        JsonElement bridge = bridgeDocument.RootElement;
        Assert.Equal("blocked-release-close-real-proof-import-required", bridge.GetProperty("validationState").GetString());
        Assert.Equal(6, bridge.GetProperty("laneCount").GetInt32());
        Assert.Equal(6, bridge.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, bridge.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(bridge.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(bridge);

        using JsonDocument checkpointDocument = ReadFinalReleaseJson("final-owner-close-readiness-checkpoint-validation.json");
        JsonElement checkpoint = checkpointDocument.RootElement;
        Assert.Equal("blocked-final-owner-close-readiness-owner-proof-required", checkpoint.GetProperty("validationState").GetString());
        int readinessCheckCount = checkpoint.GetProperty("readinessCheckCount").GetInt32();
        Assert.True(readinessCheckCount >= 12);
        Assert.Equal(readinessCheckCount, checkpoint.GetProperty("blockedReadinessCheckCount").GetInt32());
        Assert.Equal(0, checkpoint.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(checkpoint.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(checkpoint);
    }

    [Fact]
    public void ReleaseEvidenceBundleAuditAndDocsIncludePublicPublishRealProofBackfillExecution()
    {
        RunPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-public-publish-real-result-record-required", evidence.GetProperty("publicPublishRealResultRecordDraftValidationState").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", evidence.GetProperty("postPublishCleanConsumerProofRecordDraftValidationState").GetString());
        Assert.Equal("blocked-public-publish-forbidden-substitute-scan-owner-proof-required", evidence.GetProperty("publicPublishForbiddenSubstituteScanValidationState").GetString());
        Assert.Equal("blocked-release-close-real-proof-import-required", evidence.GetProperty("releaseCloseRealProofImportBridgeValidationState").GetString());
        Assert.Equal("blocked-final-owner-close-readiness-owner-proof-required", evidence.GetProperty("finalOwnerCloseReadinessCheckpointValidationState").GetString());

        Assert.Equal(12, evidence.GetProperty("publicPublishRealResultRecordDraftRequiredFieldCount").GetInt32());
        Assert.Equal(19, evidence.GetProperty("postPublishCleanConsumerProofRecordDraftRequiredFieldCount").GetInt32());
        Assert.Equal(9, evidence.GetProperty("publicPublishForbiddenSubstituteScanCheckCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("releaseCloseRealProofImportBridgeLaneCount").GetInt32());
        Assert.True(evidence.GetProperty("finalOwnerCloseReadinessCheckpointCheckCount").GetInt32() >= 12);

        Assert.False(evidence.GetProperty("publicPublishRealResultRecordDraftCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishCleanConsumerProofRecordDraftCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishForbiddenSubstituteScanCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseRealProofImportBridgeCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerCloseReadinessCheckpointCanCloseReleaseIssue").GetBoolean());

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

        Assert.Contains("articles/zh-cn/public-publish-real-result-record-draft.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-proof-record-draft.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/public-publish-forbidden-substitute-scan.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-real-proof-import-bridge.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-owner-close-readiness-checkpoint.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("public-publish-real-result-record-draft", readme, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-proof-record-draft", readmeZh, StringComparison.Ordinal);
        Assert.Contains("public-publish-forbidden-substitute-scan", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("release-close-real-proof-import-bridge", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("final-owner-close-readiness-checkpoint", releaseEvidenceDoc, StringComparison.Ordinal);
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
