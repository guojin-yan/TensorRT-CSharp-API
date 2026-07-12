using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerPublishExecutionAndArticlePlanningTests
{
    [Fact]
    public void ReplayChecklistAndEvidenceRunbookKeepOwnerPublishBlockedAndTraceable()
    {
        RunPlanningPipeline();

        using JsonDocument checklistDocument = ReadFinalReleaseJson("final-owner-publish-execution-replay-checklist-pack.json");
        JsonElement checklist = checklistDocument.RootElement;
        Assert.Equal("final-owner-publish-execution-replay-checklist-pack", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-publish-execution-replay-owner-action-required", checklist.GetProperty("checklistState").GetString());
        Assert.Equal(6, checklist.GetProperty("stepCount").GetInt32());
        Assert.Equal(6, checklist.GetProperty("blockedStepCount").GetInt32());
        AssertNonProofFlags(checklist);
        AssertReplayStep(checklist, "readonly-freeze");
        AssertReplayStep(checklist, "owner-authorization-capture");
        AssertReplayStep(checklist, "publish-command-capture");
        AssertReplayStep(checklist, "public-package-identity");
        AssertReplayStep(checklist, "post-publish-clean-consumer");
        AssertReplayStep(checklist, "rollback-and-close-readiness");

        using JsonDocument checklistValidationDocument = ReadFinalReleaseJson("final-owner-publish-execution-replay-checklist-pack-validation.json");
        JsonElement checklistValidation = checklistValidationDocument.RootElement;
        Assert.Equal("final-owner-publish-execution-replay-checklist-pack-validation-ready-non-proof", checklistValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, checklistValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(checklistValidation);

        using JsonDocument runbookDocument = ReadFinalReleaseJson("final-owner-publish-evidence-import-runbook.json");
        JsonElement runbook = runbookDocument.RootElement;
        Assert.Equal("final-owner-publish-evidence-import-runbook", runbook.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-publish-evidence-import-owner-action-required", runbook.GetProperty("runbookState").GetString());
        Assert.True(runbook.GetProperty("contractRequiredFieldCount").GetInt32() >= 178);
        Assert.Equal(6, runbook.GetProperty("sectionCount").GetInt32());
        Assert.Equal(0, runbook.GetProperty("sectionFieldMissingFromContractCount").GetInt32());
        AssertNonProofFlags(runbook);
        foreach (string section in new[] { "package-identity", "publish-transcripts", "clean-consumer", "host-runtime", "non-substitutes", "rollback-close" })
        {
            AssertRunbookSection(runbook, section);
        }

        using JsonDocument runbookValidationDocument = ReadFinalReleaseJson("final-owner-publish-evidence-import-runbook-validation.json");
        JsonElement runbookValidation = runbookValidationDocument.RootElement;
        Assert.Equal("final-owner-publish-evidence-import-runbook-validation-ready-non-proof", runbookValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, runbookValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(runbookValidation);
    }

    [Fact]
    public void PublicArticleMatrixAndPostPublishArticleProofGateBlockUnprovenClaims()
    {
        RunPlanningPipeline();

        using JsonDocument matrixDocument = ReadFinalReleaseJson("public-article-readiness-matrix.json");
        JsonElement matrix = matrixDocument.RootElement;
        Assert.Equal("public-article-readiness-matrix", matrix.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-article-proof-and-boundary-review-required", matrix.GetProperty("matrixState").GetString());
        Assert.True(matrix.GetProperty("roadmapArticleCount").GetInt32() >= 30);
        Assert.Equal(9, matrix.GetProperty("laneCount").GetInt32());
        Assert.Equal(9, matrix.GetProperty("blockedLaneCount").GetInt32());
        AssertNonProofFlags(matrix);
        foreach (string lane in new[] { "project-overview", "api-interface-guide", "source-build-cpp", "dual-package-strategy", "yolovision-series", "onnx-to-engine-trtexec", "tensorrtexec-application", "post-publish-clean-consumer", "faq-troubleshooting" })
        {
            AssertArticleLane(matrix, lane);
        }

        string matrixRaw = matrix.GetRawText();
        Assert.Contains("已发布到 NuGet", matrixRaw, StringComparison.Ordinal);
        Assert.Contains("公开包已经可安装", matrixRaw, StringComparison.Ordinal);
        Assert.Contains("release closed", matrixRaw, StringComparison.Ordinal);
        Assert.Contains("samples/YoloVision", matrixRaw, StringComparison.Ordinal);
        Assert.Contains("applications/TensorRtExec", matrixRaw, StringComparison.Ordinal);

        using JsonDocument matrixValidationDocument = ReadFinalReleaseJson("public-article-readiness-matrix-validation.json");
        JsonElement matrixValidation = matrixValidationDocument.RootElement;
        Assert.Equal("public-article-readiness-matrix-validation-ready-non-proof", matrixValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, matrixValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(matrixValidation);

        using JsonDocument proofGateDocument = ReadFinalReleaseJson("post-publish-article-proof-gate.json");
        JsonElement proofGate = proofGateDocument.RootElement;
        Assert.Equal("post-publish-article-proof-gate", proofGate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-article-proof-owner-evidence-required", proofGate.GetProperty("gateState").GetString());
        Assert.Equal(5, proofGate.GetProperty("claimGateCount").GetInt32());
        Assert.Equal(5, proofGate.GetProperty("blockedClaimGateCount").GetInt32());
        AssertNonProofFlags(proofGate);
        foreach (string claim in new[] { "nuget-published", "github-packages-published", "clean-consumer-verified", "release-closed", "runtime-proof" })
        {
            AssertClaimGate(proofGate, claim);
        }

        using JsonDocument proofGateValidationDocument = ReadFinalReleaseJson("post-publish-article-proof-gate-validation.json");
        JsonElement proofGateValidation = proofGateValidationDocument.RootElement;
        Assert.Equal("post-publish-article-proof-gate-validation-ready-non-proof", proofGateValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, proofGateValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(proofGateValidation);
    }

    private static void RunPlanningPipeline()
    {
        RunPowerShell("Export-FinalReadonlyPublishAuditPack.ps1");
        RunPowerShell("Test-FinalReadonlyPublishAuditPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerOneScreenExecutionManual.ps1");
        RunPowerShell("Test-FinalOwnerOneScreenExecutionManual.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1");
        RunPowerShell("Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1", "-Strict");
        RunPowerShell("Export-PostPublishStrictCrossCheckPack.ps1");
        RunPowerShell("Test-PostPublishStrictCrossCheckPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosure.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosure.ps1", "-Strict");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");
        RunPowerShell("Export-TechnicalArticlePublicationMatrix.ps1");
        RunPowerShell("Export-ReleaseCloseStrictProofExecutionOrder.ps1");
        RunPowerShell("Export-ArticlePublishingReadinessMap.ps1");
        RunPowerShell("Test-ArticlePublishingReadinessMap.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1");
        RunPowerShell("Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleReadinessMatrix.ps1");
        RunPowerShell("Test-PublicArticleReadinessMatrix.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerPublishEvidenceImportRunbook.ps1");
        RunPowerShell("Test-FinalOwnerPublishEvidenceImportRunbook.ps1", "-Strict");
        RunPowerShell("Export-PostPublishArticleProofGate.ps1");
        RunPowerShell("Test-PostPublishArticleProofGate.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertReplayStep(JsonElement checklist, string id)
    {
        Assert.Contains(checklist.GetProperty("steps").EnumerateArray(), step =>
            step.GetProperty("id").GetString() == id &&
            step.GetProperty("ownerActionRequired").GetBoolean() &&
            !step.GetProperty("performsPublish").GetBoolean() &&
            !step.GetProperty("isProof").GetBoolean());
    }

    private static void AssertRunbookSection(JsonElement runbook, string id)
    {
        Assert.Contains(runbook.GetProperty("sections").EnumerateArray(), section =>
            section.GetProperty("id").GetString() == id &&
            section.GetProperty("ownerActionRequired").GetBoolean() &&
            section.GetProperty("fieldCount").GetInt32() >= 5 &&
            !section.GetProperty("performsPublish").GetBoolean() &&
            !section.GetProperty("isProof").GetBoolean());
    }

    private static void AssertArticleLane(JsonElement matrix, string id)
    {
        Assert.Contains(matrix.GetProperty("lanes").EnumerateArray(), lane =>
            lane.GetProperty("id").GetString() == id &&
            lane.GetProperty("ownerProofRequired").GetBoolean() &&
            !lane.GetProperty("canPublishNow").GetBoolean() &&
            lane.GetProperty("mediaRequirementCount").GetInt32() >= 1 &&
            lane.GetProperty("codePathCount").GetInt32() >= 1);
    }

    private static void AssertClaimGate(JsonElement proofGate, string id)
    {
        Assert.Contains(proofGate.GetProperty("claimGates").EnumerateArray(), gate =>
            gate.GetProperty("id").GetString() == id &&
            gate.GetProperty("ownerActionRequired").GetBoolean() &&
            !gate.GetProperty("passed").GetBoolean() &&
            !gate.GetProperty("isProof").GetBoolean());
    }

    private static void AssertNonProofFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", scriptName), arguments);
    }
}
