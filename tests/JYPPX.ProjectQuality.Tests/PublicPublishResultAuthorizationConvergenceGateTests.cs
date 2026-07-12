using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPublishResultAuthorizationConvergenceGateTests
{
    [Fact]
    public void PublicPublishAuthorizationInputAndResultConvergenceStayBlockedNonProof()
    {
        RunPowerShell("Export-OwnerPublicPublishAuthorizationInputTemplate.ps1");
        RunPowerShell("Import-OwnerPublicPublishAuthorizationInput.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationInput.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishFinalOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPublishFinalOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishAuthorizationGate.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationGate.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishResultOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPublishResultOwnerInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPublishResultOwnerInput.ps1");
        RunPowerShell("Test-PublicPublishResultImport.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseStrictValidationBridge.ps1");
        RunPowerShell("Test-ReleaseCloseStrictValidationBridge.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishResultAuthorizationConvergenceGate.ps1");
        RunPowerShell("Test-PublicPublishResultAuthorizationConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument authorizationInputDocument = ReadFinalReleaseJson("owner-public-publish-authorization-input-validation.json");
        JsonElement authorizationInput = authorizationInputDocument.RootElement;
        Assert.Equal("owner-public-publish-authorization-input-validation", authorizationInput.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-input-required", authorizationInput.GetProperty("validationState").GetString());
        Assert.Equal(0, authorizationInput.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(authorizationInput.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(authorizationInput);

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("public-publish-result-authorization-convergence-gate.json");
        JsonElement convergence = convergenceDocument.RootElement;
        Assert.Equal("public-publish-result-authorization-convergence-gate", convergence.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-result-authorization-convergence-required", convergence.GetProperty("gateState").GetString());
        Assert.True(convergence.GetProperty("blockedGateItemCount").GetInt32() > 0);
        Assert.False(convergence.GetProperty("performsPublish").GetBoolean());
        Assert.False(convergence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(convergence.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(convergence.GetProperty("eligibleForOwnerCloseInstruction").GetBoolean());
        AssertConvergenceGateItem(convergence, "owner-authorization-input");
        AssertConvergenceGateItem(convergence, "public-publish-result-import");
        AssertConvergenceGateItem(convergence, "post-publish-clean-consumer-proof");
        AssertConvergenceGateItem(convergence, "release-close-strict-bridge");

        using JsonDocument convergenceValidationDocument = ReadFinalReleaseJson("public-publish-result-authorization-convergence-gate-validation.json");
        JsonElement convergenceValidation = convergenceValidationDocument.RootElement;
        Assert.Equal("public-publish-result-authorization-convergence-gate-validation", convergenceValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-result-authorization-convergence-required", convergenceValidation.GetProperty("validationState").GetString());
        Assert.True(convergenceValidation.GetProperty("blockedGateItemCount").GetInt32() > 0);
        Assert.Equal(0, convergenceValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(convergenceValidation.GetProperty("performsPublish").GetBoolean());
        Assert.False(convergenceValidation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(convergenceValidation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(convergenceValidation.GetProperty("eligibleForOwnerCloseInstruction").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-public-publish-authorization-input-required", evidence.GetProperty("ownerPublicPublishAuthorizationInputValidationState").GetString());
        Assert.Equal("blocked-public-publish-result-authorization-convergence-required", evidence.GetProperty("publicPublishResultAuthorizationConvergenceGateValidationState").GetString());
        Assert.False(evidence.GetProperty("ownerPublicPublishAuthorizationInputCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerPublicPublishAuthorizationInputCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishResultAuthorizationConvergenceGateCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishResultAuthorizationConvergenceGateCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishResultAuthorizationConvergenceGateEligibleForOwnerCloseInstruction").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "owner-public-publish-authorization-input", "not package push");
        AssertBlockedEvidenceItem(evidence, "public-publish-result-authorization-convergence-gate", "not owner close instruction");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input.template.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input-import.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input-import.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-input-validation.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-authorization-convergence-gate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-authorization-convergence-gate.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-authorization-convergence-gate-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-authorization-convergence-gate-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "owner-public-publish-authorization-input");
        AssertAuditedNonProofItem(audit, "public-publish-result-authorization-convergence-gate");
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

    private static void AssertConvergenceGateItem(JsonElement convergence, string id)
    {
        Assert.Contains(convergence.GetProperty("gateItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("ready").GetBoolean() == false);
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

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
