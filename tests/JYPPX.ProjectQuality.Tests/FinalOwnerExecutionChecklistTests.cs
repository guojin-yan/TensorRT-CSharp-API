using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionChecklistTests
{
    [Fact]
    public void ChecklistPrescribesFinalOwnerPathWithoutAutomatedPush()
    {
        OwnerRealInputLandingPackTests.RunPowerShell("Export-FinalOwnerExecutionChecklist.ps1");
        OwnerRealInputLandingPackTests.RunPowerShell("Test-FinalOwnerExecutionChecklist.ps1", "-Strict");

        using JsonDocument checklistDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("final-owner-execution-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("final-owner-execution-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-checklist-real-owner-input-required", checklist.GetProperty("checklistState").GetString());
        Assert.Equal(6, checklist.GetProperty("stepCount").GetInt32());
        Assert.Equal(6, checklist.GetProperty("blockedStepCount").GetInt32());
        OwnerRealInputLandingPackTests.AssertFlagsStayNonProof(checklist);
        OwnerRealInputLandingPackTests.AssertBoundary(checklist.GetProperty("boundary").GetString()!);
        Assert.DoesNotContain("dotnet nuget push", checklist.GetProperty("commands").GetRawText(), StringComparison.OrdinalIgnoreCase);

        string[] stepIds = checklist.GetProperty("executionSteps").EnumerateArray().Select(static item => item.GetProperty("stepId").GetString()!).ToArray();
        Assert.Contains("01-owner-authorization", stepIds);
        Assert.Contains("02-clean-external-package-consumer", stepIds);
        Assert.Contains("03-linux-runner-proof", stepIds);
        Assert.Contains("04-real-model-runtime", stepIds);
        Assert.Contains("05-post-publish-verification", stepIds);
        Assert.Contains("06-final-close-validation", stepIds);

        foreach (JsonElement step in checklist.GetProperty("executionSteps").EnumerateArray())
        {
            Assert.True(step.GetProperty("requiredBackfillFields").GetArrayLength() >= 7);
            Assert.True(step.GetProperty("expectedArtifacts").GetArrayLength() >= 1);
            Assert.Contains("Test-", step.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
            Assert.Contains("stdout", step.GetProperty("requiredCapture").GetRawText(), StringComparison.Ordinal);
            Assert.Contains("stderr", step.GetProperty("requiredCapture").GetRawText(), StringComparison.Ordinal);
            Assert.Contains("SHA256", step.GetProperty("requiredCapture").GetRawText(), StringComparison.Ordinal);
            Assert.Contains("host identity", step.GetProperty("requiredCapture").GetRawText(), StringComparison.Ordinal);
            OwnerRealInputLandingPackTests.AssertFlagsStayNonProof(step);
            OwnerRealInputLandingPackTests.AssertBoundary(step.GetProperty("boundary").GetString()!);
        }

        using JsonDocument validationDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("final-owner-execution-checklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-final-owner-execution-checklist-real-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        OwnerRealInputLandingPackTests.AssertFlagsStayNonProof(validation);
        OwnerRealInputLandingPackTests.AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void ReleaseEvidenceCarriesChecklistAsRequiredNonProofItem()
    {
        OwnerRealInputLandingPackTests.RunOwnerInputPipeline();

        using JsonDocument evidenceDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-final-owner-execution-checklist-real-owner-input-required", evidence.GetProperty("finalOwnerExecutionChecklistValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("finalOwnerExecutionChecklistStepCount").GetInt32());
        Assert.False(evidence.GetProperty("finalOwnerExecutionChecklistCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerExecutionChecklistCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerExecutionChecklistIsRuntimeExecutionProof").GetBoolean());

        OwnerRealInputLandingPackTests.AssertBlockedEvidenceItem(evidence, "final-owner-execution-checklist");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/final-owner-execution-checklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-owner-execution-checklist-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = OwnerRealInputLandingPackTests.ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        OwnerRealInputLandingPackTests.AssertAuditedNonProofItem(audit, "final-owner-execution-checklist");
    }
}
