using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerStrictCloseExecutionOrderTests
{
    [Fact]
    public void ExecutionOrderAggregatesOwnerStepsWithoutPromotingProof()
    {
        RunPipeline();

        using JsonDocument orderDocument = ReadFinalReleaseJson("final-owner-strict-close-execution-order.json");
        JsonElement order = orderDocument.RootElement;

        Assert.Equal("final-owner-strict-close-execution-order", order.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-strict-close-owner-execution-required", order.GetProperty("orderState").GetString());
        Assert.Equal(7, order.GetProperty("stepCount").GetInt32());
        Assert.Equal(7, order.GetProperty("blockedStepCount").GetInt32());
        Assert.True(order.GetProperty("sourceRecordCount").GetInt32() >= 9);
        Assert.True(order.GetProperty("sourceActionCount").GetInt32() >= 7);
        Assert.True(order.GetProperty("sourceExecutionStepCount").GetInt32() >= 7);
        Assert.True(order.GetProperty("cleanExternalRunbookStepCount").GetInt32() >= 9);
        Assert.True(order.GetProperty("postPublishRunbookStepCount").GetInt32() >= 6);
        Assert.True(order.GetProperty("publicPublishExecutionLaneCount").GetInt32() >= 10);
        Assert.True(order.GetProperty("publicPublishCrossCheckCount").GetInt32() >= 11);
        Assert.True(order.GetProperty("finalOwnerCloseReadinessCheckCount").GetInt32() >= 12);
        Assert.True(order.GetProperty("finalReleaseCloseBlockerCount").GetInt32() >= 19);
        Assert.True(order.GetProperty("ownerInputContractSurfaceCount").GetInt32() >= 5);
        Assert.True(order.GetProperty("ownerInputContractCanonicalFieldCount").GetInt32() >= 14);
        Assert.Equal(2, order.GetProperty("ownerInputContractRunbookInputCount").GetInt32());
        Assert.True(order.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(order.GetProperty("performsPublish").GetBoolean());
        Assert.False(order.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(order.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(order.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(order.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(order.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(order.GetProperty("isReleaseCloseProof").GetBoolean());
        AssertBoundary(order.GetProperty("boundary").GetString()!);

        JsonElement[] steps = order.GetProperty("executionSteps").EnumerateArray().ToArray();
        Assert.Equal(7, steps.Length);
        Assert.All(steps, step =>
        {
            Assert.False(string.IsNullOrWhiteSpace(step.GetProperty("id").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(step.GetProperty("title").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(step.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(step.GetProperty("blockedUntil").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(step.GetProperty("failureStopRule").GetString()));
            Assert.True(step.GetProperty("inputArtifactCount").GetInt32() >= 1);
            Assert.True(step.GetProperty("outputArtifactCount").GetInt32() >= 1);
            Assert.True(step.GetProperty("validatorScriptCount").GetInt32() >= 1);
            Assert.True(step.GetProperty("notExecutedByAutomation").GetBoolean());
            Assert.False(step.GetProperty("performsPublish").GetBoolean());
            Assert.False(step.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(step.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(step.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(step.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(step.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(step.GetProperty("isReleaseCloseProof").GetBoolean());
            AssertBoundary(step.GetProperty("boundary").GetString()!);
        });

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-strict-close-execution-order-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("final-owner-strict-close-execution-order-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-strict-close-owner-execution-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(7, validation.GetProperty("stepCount").GetInt32());
        Assert.Equal(7, validation.GetProperty("blockedStepCount").GetInt32());
        Assert.True(validation.GetProperty("sourceRecordCount").GetInt32() >= 9);
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void EvidenceBundleAndClassificationAuditIncludeExecutionOrderAsNonProof()
    {
        RunPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-final-owner-strict-close-owner-execution-required", evidence.GetProperty("finalOwnerStrictCloseExecutionOrderState").GetString());
        Assert.Equal("blocked-final-owner-strict-close-owner-execution-required", evidence.GetProperty("finalOwnerStrictCloseExecutionOrderValidationState").GetString());
        Assert.Equal(7, evidence.GetProperty("finalOwnerStrictCloseExecutionOrderStepCount").GetInt32());
        Assert.Equal(7, evidence.GetProperty("finalOwnerStrictCloseExecutionOrderBlockedStepCount").GetInt32());
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderSourceRecordCount").GetInt32() >= 9);
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderReadinessCheckCount").GetInt32() >= 12);
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderBlockerCount").GetInt32() >= 19);
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderContractSurfaceCount").GetInt32() >= 5);
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderCanonicalFieldCount").GetInt32() >= 14);
        Assert.Equal(2, evidence.GetProperty("finalOwnerStrictCloseExecutionOrderRunbookInputCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("finalOwnerStrictCloseExecutionOrderFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderNotExecutedByAutomation").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderPerformsPublish").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderIsPostPublishProof").GetBoolean());
        Assert.False(evidence.GetProperty("finalOwnerStrictCloseExecutionOrderIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "final-owner-strict-close-execution-order");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("steps=7", evidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        AssertBoundary(evidenceItem.GetProperty("boundary").GetString()!);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/final-owner-strict-close-execution-order.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-owner-strict-close-execution-order.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-owner-strict-close-execution-order-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-owner-strict-close-execution-order-validation.md", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == "final-owner-strict-close-execution-order" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-OwnerInputContractConvergence.ps1");
        RunPowerShell("Test-OwnerInputContractConvergence.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerStrictCloseExecutionOrder.ps1");
        RunPowerShell("Test-FinalOwnerStrictCloseExecutionOrder.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
