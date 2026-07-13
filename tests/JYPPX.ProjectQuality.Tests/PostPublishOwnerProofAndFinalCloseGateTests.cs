using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishOwnerProofAndFinalCloseGateTests
{
    private static readonly string[] NewEvidenceIds =
    [
        "public-publish-real-result-owner-input-contract",
        "post-publish-clean-consumer-proof-record-contract",
        "release-issue-close-strict-owner-decision-import",
        "final-close-gate-convergence",
    ];

    private static readonly string[] NewSourceArtifacts =
    [
        "artifacts/final-release/public-publish-real-result-owner-input-contract.json",
        "artifacts/final-release/public-publish-real-result-owner-input-contract.md",
        "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
        "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.md",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.md",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
        "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.md",
        "artifacts/final-release/release-issue-close-strict-owner-decision-import.json",
        "artifacts/final-release/release-issue-close-strict-owner-decision-import.md",
        "artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json",
        "artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.md",
        "artifacts/final-release/final-close-gate-convergence.json",
        "artifacts/final-release/final-close-gate-convergence.md",
        "artifacts/final-release/final-close-gate-convergence-validation.json",
        "artifacts/final-release/final-close-gate-convergence-validation.md",
    ];

    [Fact]
    public void PostPublishOwnerProofContractsStayBlockedNonProof()
    {
        RunPostPublishOwnerProofPipeline();

        using JsonDocument publishContractDocument = ReadFinalReleaseJson("public-publish-real-result-owner-input-contract-validation.json");
        JsonElement publishContract = publishContractDocument.RootElement;
        Assert.Equal("blocked-public-publish-real-result-owner-input-required", publishContract.GetProperty("validationState").GetString());
        Assert.Equal(12, publishContract.GetProperty("requiredFieldCount").GetInt32());
        Assert.Equal(12, publishContract.GetProperty("blockedRequiredFieldCount").GetInt32());
        Assert.Equal(0, publishContract.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(publishContract.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(publishContract);

        using JsonDocument cleanConsumerDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-record-contract-validation.json");
        JsonElement cleanConsumer = cleanConsumerDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", cleanConsumer.GetProperty("validationState").GetString());
        int cleanConsumerRequiredFieldCount = cleanConsumer.GetProperty("requiredFieldCount").GetInt32();
        Assert.True(cleanConsumerRequiredFieldCount >= 46);
        Assert.Equal(cleanConsumerRequiredFieldCount, cleanConsumer.GetProperty("blockedRequiredFieldCount").GetInt32());
        Assert.Equal(0, cleanConsumer.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(cleanConsumer.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(cleanConsumer);

        using JsonDocument decisionImportDocument = ReadFinalReleaseJson("release-issue-close-strict-owner-decision-import-validation.json");
        JsonElement decisionImport = decisionImportDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-strict-owner-decision-required", decisionImport.GetProperty("validationState").GetString());
        Assert.Equal(5, decisionImport.GetProperty("laneCount").GetInt32());
        Assert.Equal(5, decisionImport.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, decisionImport.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(decisionImport.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(decisionImport);

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("final-close-gate-convergence-validation.json");
        JsonElement convergence = convergenceDocument.RootElement;
        Assert.Equal("blocked-final-close-gate-owner-proof-required", convergence.GetProperty("validationState").GetString());
        int convergenceLaneCount = convergence.GetProperty("laneCount").GetInt32();
        Assert.True(convergenceLaneCount >= 10);
        Assert.Equal(convergenceLaneCount, convergence.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(0, convergence.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(convergence.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(convergence);
    }

    [Fact]
    public void ReleaseEvidenceAuditAndDocsIncludePostPublishOwnerProofContracts()
    {
        RunPostPublishOwnerProofPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-public-publish-real-result-owner-input-required", evidence.GetProperty("publicPublishRealResultOwnerInputContractValidationState").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", evidence.GetProperty("postPublishCleanConsumerProofRecordContractValidationState").GetString());
        Assert.Equal("blocked-release-issue-close-strict-owner-decision-required", evidence.GetProperty("releaseIssueCloseStrictOwnerDecisionImportValidationState").GetString());
        Assert.Equal("blocked-final-close-gate-owner-proof-required", evidence.GetProperty("finalCloseGateConvergenceValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPublishRealResultOwnerInputContractCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishCleanConsumerProofRecordContractCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseStrictOwnerDecisionImportCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalCloseGateConvergenceCanCloseReleaseIssue").GetBoolean());

        foreach (string id in NewEvidenceIds)
        {
            AssertBlockedEvidenceItem(evidence, id);
        }

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string sourceArtifact in NewSourceArtifacts)
        {
            Assert.Contains(sourceArtifact, sourceArtifacts);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string id in NewEvidenceIds)
        {
            AssertAuditedNonProofItem(audit, id);
        }

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/public-publish-real-result-owner-input-contract.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-proof-record-contract.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-strict-owner-decision-import.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-close-gate-convergence.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("public-publish-real-result-owner-input-contract", readme, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-proof-record-contract", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-strict-owner-decision-import", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("final-close-gate-convergence", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    private static void RunPostPublishOwnerProofPipeline()
    {
        RunPowerShell("Export-PublicPublishRealResultOwnerInputContract.ps1");
        RunPowerShell("Test-PublicPublishRealResultOwnerInputContract.ps1", "-Strict");
        RunPowerShell("Export-PostPublishCleanConsumerProofRecordContract.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofRecordContract.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseStrictOwnerDecisionImport.ps1");
        RunPowerShell("Test-ReleaseIssueCloseStrictOwnerDecisionImport.ps1", "-Strict");
        RunPowerShell("Export-FinalCloseGateConvergence.ps1");
        RunPowerShell("Test-FinalCloseGateConvergence.ps1", "-Strict");
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
