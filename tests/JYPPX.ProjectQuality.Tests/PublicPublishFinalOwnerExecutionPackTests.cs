using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPublishFinalOwnerExecutionPackTests
{
    private static readonly JsonSerializerOptions IndentedJsonOptions = new()
    {
        WriteIndented = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
    };

    [Fact]
    public void FinalOwnerExecutionPackArtifactsStayBlockedNonProof()
    {
        RunFinalOwnerExecutionPipeline();

        using JsonDocument executionPackDocument = ReadFinalReleaseJson("public-publish-final-owner-execution-pack-validation.json");
        JsonElement executionPack = executionPackDocument.RootElement;
        Assert.Equal("blocked-public-publish-final-owner-execution-required", executionPack.GetProperty("validationState").GetString());
        Assert.Equal(10, executionPack.GetProperty("executionLaneCount").GetInt32());
        Assert.Equal(10, executionPack.GetProperty("blockedExecutionLaneCount").GetInt32());
        Assert.Equal(0, executionPack.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(executionPack);

        using JsonDocument crossCheckDocument = ReadFinalReleaseJson("public-publish-command-cross-check-validation.json");
        JsonElement crossCheck = crossCheckDocument.RootElement;
        Assert.Equal("blocked-public-publish-command-cross-check-owner-action-required", crossCheck.GetProperty("validationState").GetString());
        Assert.Equal(11, crossCheck.GetProperty("crossCheckCount").GetInt32());
        Assert.Equal(11, crossCheck.GetProperty("blockedCrossCheckCount").GetInt32());
        Assert.Equal(0, crossCheck.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(crossCheck);

        using JsonDocument decisionDocument = ReadFinalReleaseJson("release-issue-close-owner-decision-input-validation.json");
        JsonElement decision = decisionDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-owner-decision-input-required", decision.GetProperty("validationState").GetString());
        Assert.Equal(0, decision.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(decision.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.Equal("blocked-final-public-release-closure-real-owner-proof-required", decision.GetProperty("finalPublicReleaseClosureBridgeValidationState").GetString());
        Assert.Equal(9, decision.GetProperty("closureLaneCount").GetInt32());
        Assert.True(decision.GetProperty("closureBlockedLaneCount").GetInt32() > 0);
        Assert.False(decision.GetProperty("postPublishProofCandidateReady").GetBoolean());
        Assert.False(decision.GetProperty("postPublishProofSourceLinkageReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(decision);

        string[] decisionValidationItemIds = decision.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("final-public-release-closure-bridge-hash", decisionValidationItemIds);
        Assert.Contains("final-bridge-lane-count-match", decisionValidationItemIds);
        Assert.Contains("final-bridge-ready-before-close", decisionValidationItemIds);
        Assert.Contains("post-publish-source-proof-linkage-ready", decisionValidationItemIds);
        Assert.Contains("public-package-sha-match", decisionValidationItemIds);

        using JsonDocument freezeAuditDocument = ReadFinalReleaseJson("final-evidence-freeze-non-proof-audit-validation.json");
        JsonElement freezeAudit = freezeAuditDocument.RootElement;
        Assert.Equal("blocked-final-evidence-freeze-non-proof-audit", freezeAudit.GetProperty("validationState").GetString());
        Assert.True(freezeAudit.GetProperty("auditLaneCount").GetInt32() >= 8);
        Assert.Equal(0, freezeAudit.GetProperty("boundaryFailureCount").GetInt32());
        Assert.Equal(0, freezeAudit.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(freezeAudit);

        using JsonDocument authorizationGateDocument = ReadFinalReleaseJson("owner-public-publish-authorization-gate-validation.json");
        JsonElement authorizationGate = authorizationGateDocument.RootElement;
        Assert.Equal("blocked-owner-public-publish-authorization-required", authorizationGate.GetProperty("validationState").GetString());
        Assert.True(authorizationGate.GetProperty("blockedActionRequiredCount").GetInt32() > 0);
        Assert.Equal(0, authorizationGate.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(authorizationGate);
    }

    [Fact]
    public void ReleaseEvidenceAuditAndDocsIncludeFinalOwnerExecutionPack()
    {
        RunFinalOwnerExecutionPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-public-publish-final-owner-execution-required", evidence.GetProperty("publicPublishFinalOwnerExecutionPackValidationState").GetString());
        Assert.Equal("blocked-public-publish-command-cross-check-owner-action-required", evidence.GetProperty("publicPublishCommandCrossCheckValidationState").GetString());
        Assert.Equal("blocked-release-issue-close-owner-decision-input-required", evidence.GetProperty("releaseIssueCloseOwnerDecisionInputValidationState").GetString());
        Assert.Equal("blocked-final-evidence-freeze-non-proof-audit", evidence.GetProperty("finalEvidenceFreezeNonProofAuditValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPublishFinalOwnerExecutionPackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishCommandCrossCheckCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseOwnerDecisionInputCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalEvidenceFreezeNonProofAuditCanCloseReleaseIssue").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "public-publish-final-owner-execution-pack");
        AssertBlockedEvidenceItem(evidence, "public-publish-command-cross-check");
        AssertBlockedEvidenceItem(evidence, "release-issue-close-owner-decision-input");
        AssertBlockedEvidenceItem(evidence, "final-evidence-freeze-non-proof-audit");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/public-publish-final-owner-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-final-owner-execution-pack.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-final-owner-execution-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-final-owner-execution-pack-validation.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-command-cross-check.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-command-cross-check.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-command-cross-check-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-command-cross-check-validation.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input.template.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input-validation.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze-non-proof-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze-non-proof-audit.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze-non-proof-audit-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-evidence-freeze-non-proof-audit-validation.md", sourceArtifacts);

        using JsonDocument authorizationGateDocument = ReadFinalReleaseJson("owner-public-publish-authorization-gate.json");
        JsonElement authorizationGate = authorizationGateDocument.RootElement;
        Assert.Equal("owner-public-publish-authorization-gate", authorizationGate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-required", authorizationGate.GetProperty("gateState").GetString());
        Assert.Equal(string.Empty, authorizationGate.GetProperty("materializedExecutableCommand").GetString());
        Assert.Contains("never executes dotnet nuget push", authorizationGate.GetProperty("safetyBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "public-publish-final-owner-execution-pack");
        AssertAuditedNonProofItem(audit, "public-publish-command-cross-check");
        AssertAuditedNonProofItem(audit, "release-issue-close-owner-decision-input");
        AssertAuditedNonProofItem(audit, "final-evidence-freeze-non-proof-audit");

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/public-publish-final-owner-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/public-publish-command-cross-check.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-owner-proof-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-owner-decision-input.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-evidence-freeze-non-proof-audit.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("public-publish-final-owner-execution-pack", readme, StringComparison.Ordinal);
        Assert.Contains("public-publish-command-cross-check", readmeZh, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-owner-proof-input", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("final-evidence-freeze-non-proof-audit", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseIssueCloseOwnerDecisionRejectsApprovedCloseWhenFinalBridgeIsStillBlocked()
    {
        RunPowerShell("Export-FinalPublicReleaseClosureBridge.ps1");
        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");

        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-issue-close-owner-decision-input.template.json");
        JsonObject input = JsonNode.Parse(File.ReadAllText(templatePath))!.AsObject();
        input["ownerDecisionInputState"] = "owner-filled-release-issue-close-owner-decision-input";
        input["ownerName"] = "owner";
        input["ownerEmail"] = "owner@example.com";
        input["ownerDecisionTimestampUtc"] = DateTimeOffset.UtcNow.ToString("O");
        input["releaseIssueUrl"] = "https://github.com/guojin-yan/TensorRT-CSharp-API/issues/1";
        input["releaseIssueNumber"] = "1";
        input["selectedChannel"] = "nuget.org";
        input["approvedPublicPackageProofHash"] = new string('a', 64);
        input["approvedPostPublishProofHash"] = new string('b', 64);
        input["rollbackPlan"] = "Owner will unlist packages and republish after blocker repair.";
        input["rollbackOwner"] = "owner";
        input["rollbackTrigger"] = "Critical post-publish runtime regression.";
        input["knownLimitationsAcknowledgement"] = "Known limitations reviewed.";
        input["finalCloseDecision"] = "approved-close-release-issue";
        input["closureLaneCount"] = 99;

        string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-issue-close-owner-decision-input.misuse.json");
        File.WriteAllText(misusePath, input.ToJsonString(IndentedJsonOptions));

        RunPowerShell(
            "Test-ReleaseIssueCloseOwnerDecisionInput.ps1",
            "-InputPath",
            "artifacts/final-release/release-issue-close-owner-decision-input.misuse.json");

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-issue-close-owner-decision-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("invalid-release-issue-close-owner-decision-input", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        AssertValidationItemFailed(validation, "approved-close-requires-ready-final-bridge", "blocker");
        AssertValidationItemFailed(validation, "approved-close-requires-post-publish-source-linkage", "blocker");
        AssertValidationItemFailed(validation, "final-bridge-lane-count-match", "blocker");
    }

    private static void RunFinalOwnerExecutionPipeline()
    {
        RunPowerShell("Export-PublicPublishFinalOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPublishFinalOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishCommandCrossCheck.ps1");
        RunPowerShell("Test-PublicPublishCommandCrossCheck.ps1", "-Strict");
        RunPowerShell("Export-FinalPublicReleaseClosureBridge.ps1");
        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");
        RunPowerShell("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", "-Strict");
        RunPowerShell("Export-FinalEvidenceFreezeNonProofAudit.ps1");
        RunPowerShell("Test-FinalEvidenceFreezeNonProofAudit.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishAuthorizationGate.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationGate.ps1", "-Strict");
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

    private static void AssertValidationItemFailed(JsonElement validation, string id, string severity)
    {
        JsonElement item = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Single(candidate => candidate.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Equal(severity, item.GetProperty("severity").GetString());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
