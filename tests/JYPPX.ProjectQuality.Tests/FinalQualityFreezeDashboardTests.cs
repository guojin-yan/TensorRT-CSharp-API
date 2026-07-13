using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalQualityFreezeDashboardTests
{
    [Fact]
    public void DashboardStaysBlockedAndNonProof()
    {
        RunPipeline();

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-quality-freeze-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;

        Assert.Equal("final-quality-freeze-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-quality-freeze-real-proof-required", dashboard.GetProperty("freezeState").GetString());
        Assert.True(dashboard.GetProperty("inputRecordCount").GetInt32() >= 21);
        Assert.True(dashboard.GetProperty("ownerRealInputControlRecordCount").GetInt32() >= 14);
        Assert.Equal(0, dashboard.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(dashboard.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(dashboard);
        AssertBoundary(dashboard.GetProperty("boundary").GetString()!);

        string[] inputIds = dashboard.GetProperty("inputRecords")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        foreach (string id in RequiredOwnerRealInputControlIds)
        {
            Assert.Contains(id, inputIds);
        }

        string[] dashboardSourceArtifacts = dashboard.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts\\final-release\\owner-real-input-json-contract-validation.json", dashboardSourceArtifacts);
        Assert.Contains("artifacts\\final-release\\public-package-hash-cross-check-gate-validation.json", dashboardSourceArtifacts);
        Assert.Contains("artifacts\\final-release\\clean-consumer-runtime-proof-cross-check-gate-validation.json", dashboardSourceArtifacts);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-quality-freeze-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("final-quality-freeze-dashboard-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-quality-freeze-real-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("findingCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditCarryDashboardAsNonProof()
    {
        RunPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-final-quality-freeze-real-proof-required", evidence.GetProperty("finalQualityFreezeDashboardState").GetString());
        Assert.Equal("blocked-final-quality-freeze-real-proof-required", evidence.GetProperty("finalQualityFreezeDashboardValidationState").GetString());
        Assert.True(evidence.GetProperty("finalQualityFreezeDashboardInputRecordCount").GetInt32() >= 21);
        Assert.Equal(0, evidence.GetProperty("finalQualityFreezeDashboardFindingCount").GetInt32());
        Assert.False(evidence.GetProperty("finalQualityFreezeDashboardCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("finalQualityFreezeDashboardCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalQualityFreezeDashboardIsRuntimeExecutionProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "final-quality-freeze-dashboard");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/final-quality-freeze-dashboard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-quality-freeze-dashboard-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-quality-freeze-dashboard" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    internal static void RunPipeline()
    {
        RunPowerShell("Export-ReleaseCandidateRealProofFinalFreeze.ps1");
        RunPowerShell("Test-ReleaseCandidateRealProofFinalFreeze.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputImportPreflight.ps1");
        RunPowerShell("Test-OwnerRealInputImportPreflight.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageHashCrossCheckGate.ps1");
        RunPowerShell("Test-PublicPackageHashCrossCheckGate.ps1", "-Strict");
        RunPowerShell("Export-CleanConsumerRuntimeProofCrossCheckGate.ps1");
        RunPowerShell("Test-CleanConsumerRuntimeProofCrossCheckGate.ps1", "-Strict");
        RunPowerShell("Export-PostPublishRollbackOwnerDecisionGate.ps1");
        RunPowerShell("Test-PostPublishRollbackOwnerDecisionGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalRealInputAdmissionPack.ps1");
        RunPowerShell("Test-ReleaseCloseFinalRealInputAdmissionPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputJsonContract.ps1");
        RunPowerShell("Test-OwnerRealInputJsonContract.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputJsonImport.ps1");
        RunPowerShell("Test-OwnerRealInputJsonImport.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputHashAndPathValidator.ps1");
        RunPowerShell("Test-OwnerRealInputHashAndPathValidator.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputForbiddenSubstituteValidator.ps1");
        RunPowerShell("Test-OwnerRealInputForbiddenSubstituteValidator.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseRealInputDryRun.ps1");
        RunPowerShell("Test-StrictCloseRealInputDryRun.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseRealInputFindingReport.ps1");
        RunPowerShell("Test-StrictCloseRealInputFindingReport.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseOwnerActionPack.ps1");
        RunPowerShell("Test-StrictCloseOwnerActionPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseRealInputFinalBlockerLedger.ps1");
        RunPowerShell("Test-ReleaseCloseRealInputFinalBlockerLedger.ps1", "-Strict");
        RunPowerShell("Export-RealOwnerEvidenceStrictValidatorOrchestration.ps1");
        RunPowerShell("Test-RealOwnerEvidenceStrictValidatorOrchestration.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseRealInputCandidatePromotionReadiness.ps1");
        RunPowerShell("Test-ReleaseCloseRealInputCandidatePromotionReadiness.ps1", "-Strict");
        RunPowerShell("Test-FinalPublishProofGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-FinalQualityFreezeDashboard.ps1");
        RunPowerShell("Test-FinalQualityFreezeDashboard.ps1", "-Strict");
    }

    private static readonly string[] RequiredOwnerRealInputControlIds =
    [
        "release-candidate-real-proof-final-freeze",
        "owner-real-input-import-preflight",
        "public-package-hash-cross-check-gate",
        "clean-consumer-runtime-proof-cross-check-gate",
        "post-publish-rollback-owner-decision-gate",
        "release-close-final-real-input-admission-pack",
        "owner-real-input-json-contract",
        "owner-real-input-json-import",
        "owner-real-input-hash-and-path-validator",
        "owner-real-input-forbidden-substitute-validator",
        "strict-close-real-input-dry-run",
        "strict-close-real-input-finding-report",
        "strict-close-owner-action-pack",
        "release-close-real-input-final-blocker-ledger",
    ];

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

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", scriptName), arguments);
    }
}
