using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseEvidenceNewProofInputsTests
{
    [Fact]
    public void ReleaseEvidenceBundleConsumesReferenceBridgeAndShardCoverageWithoutPromotingRelease()
    {
        using JsonDocument shardCoverageDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-shard-class-coverage.json")));
        JsonElement shardCoverageRoot = shardCoverageDocument.RootElement;

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        Assert.Contains("Release evidence bundle written", output, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-evidence-bundle.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("release-evidence-bundle", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());

        Assert.Equal("verified-local-source-owner-review-required", root.GetProperty("yoloVisionReferenceAssetAcquisitionState").GetString());
        Assert.True(root.GetProperty("yoloVisionReferenceAssetFilesReady").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetLicensesReady").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetCanPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetCanPublishPublicly").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("yoloVisionReferenceAssetLicenseApprovalValidationState").GetString());
        Assert.Equal(0, root.GetProperty("yoloVisionReferenceAssetLicenseApprovalFailedBlockerCount").GetInt32());
        Assert.Equal(22, root.GetProperty("yoloVisionReferenceAssetLicenseApprovalOwnerActionRequiredCount").GetInt32());
        Assert.True(root.GetProperty("yoloVisionReferenceAssetLicenseApprovalAllHashesMatchAcquisitionReport").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetLicenseApprovalAllAssetsApproved").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetLicenseApprovalCanPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetLicenseApprovalCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("yoloVisionReferenceAssetLicenseApprovalCanCloseReleaseIssue").GetBoolean());

        Assert.Equal("compatible-host-bridge-package-runtime", root.GetProperty("trt10BridgeRuntimeConsumerClassification").GetString());
        Assert.Equal("passed", root.GetProperty("trt10BridgeRuntimeConsumerSmokeStatus").GetString());
        Assert.True(root.GetProperty("trt10BridgeRuntimeConsumerIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("trt10BridgeRuntimeConsumerIsPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("trt10BridgeRuntimeConsumerCanPromoteRuntimeProof").GetBoolean());

        Assert.Equal("compatible-host-bridge-package-runtime-failed", root.GetProperty("trt11BridgeRuntimeConsumerClassification").GetString());
        Assert.Equal("failed", root.GetProperty("trt11BridgeRuntimeConsumerSmokeStatus").GetString());
        Assert.Equal(1, root.GetProperty("trt11BridgeRuntimeConsumerExitCode").GetInt32());
        Assert.False(root.GetProperty("trt11BridgeRuntimeConsumerIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("trt11BridgeRuntimeConsumerIsPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("trt11BridgeRuntimeConsumerCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("classified-owner-action-required", root.GetProperty("trt11RootCauseReportState").GetString());
        Assert.Equal("createInferRuntime-null", root.GetProperty("trt11RootCauseFailureSignature").GetString());
        Assert.Equal("trt11-create-runtime-null-cuda-runtime-error", root.GetProperty("trt11RootCauseCategory").GetString());
        Assert.True(root.GetProperty("trt11RootCauseTensorRtAvailable").GetBoolean());
        Assert.True(root.GetProperty("trt11RootCauseCudaAvailable").GetBoolean());
        Assert.True(root.TryGetProperty("trt11RootCauseNativeCreateRuntimeDiagnosticAvailable", out _));
        Assert.True(root.TryGetProperty("trt11RootCauseNativeCreateRuntimeAttempted", out _));
        Assert.True(root.TryGetProperty("trt11RootCauseNativeCreateRuntimeReturnedNull", out _));
        Assert.True(root.TryGetProperty("trt11RootCauseNativeCreateRuntimeLastStatus", out _));
        Assert.Equal("after-createInferRuntime-null-guard-ok", root.GetProperty("trt11RootCauseNativeCreateRuntimePhase").GetString());
        Assert.True(int.Parse(root.GetProperty("trt11RootCauseNativeCreateRuntimeLoggerMessageCount").GetString()!) >= 1);
        Assert.Matches("Cuda Runtime|catchCudaError", root.GetProperty("trt11RootCauseNativeCreateRuntimeLastLoggerMessage").GetString()!);
        Assert.False(root.GetProperty("trt11RootCauseCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("trt11RootCauseIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("trt11RootCauseIsPackageConsumerRuntimeProof").GetBoolean());
        Assert.StartsWith("diagnostic-", root.GetProperty("trt11DllResolutionReportState").GetString(), StringComparison.Ordinal);
        Assert.True(root.GetProperty("trt11DllResolutionMissingRequiredDllGroupCount").GetInt32() >= 0);
        Assert.True(root.GetProperty("trt11DllResolutionDuplicateDllGroupCount").GetInt32() >= 0);
        Assert.True(root.GetProperty("trt11DllResolutionOwnerActionRequired").GetBoolean());
        Assert.False(root.GetProperty("trt11DllResolutionCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("trt11DllResolutionIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("trt11DllResolutionIsPackageConsumerRuntimeProof").GetBoolean());
        Assert.Equal("diagnostic-diff-ready-non-proof", root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffState").GetString());
        Assert.True(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffDifferingFieldCount").GetInt32() >= 10);
        Assert.True(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt10SmokePassed").GetBoolean());
        Assert.True(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt11SmokeFailed").GetBoolean());
        Assert.Equal("createInferRuntime-null", root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffFailureSignature").GetString());
        Assert.False(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffIsPackageConsumerRuntimeProof").GetBoolean());
        Assert.Equal("blocked-clean-public-package-consumer-proof-owner-action-required", root.GetProperty("cleanPublicPackageConsumerProofGapReportState").GetString());
        Assert.Equal(6, root.GetProperty("cleanPublicPackageConsumerProofGapCount").GetInt32());
        Assert.Equal(6, root.GetProperty("cleanPublicPackageConsumerProofOwnerActionRequiredCount").GetInt32());
        Assert.True(root.GetProperty("cleanPublicPackageConsumerProofOwnerInputFailedActionRequiredCount").GetInt32() >= 31);
        Assert.False(root.GetProperty("cleanPublicPackageConsumerProofCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("cleanPublicPackageConsumerProofCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("cleanPublicPackageConsumerProofCanCloseReleaseIssue").GetBoolean());

        Assert.Equal(shardCoverageRoot.GetProperty("coverageState").GetString(), root.GetProperty("projectQualityShardCoverageState").GetString());
        Assert.Equal(shardCoverageRoot.GetProperty("inventoryClassCount").GetInt32(), root.GetProperty("projectQualityShardCoverageInventoryClassCount").GetInt32());
        Assert.Equal(shardCoverageRoot.GetProperty("coveredClassCount").GetInt32(), root.GetProperty("projectQualityShardCoverageCoveredClassCount").GetInt32());
        Assert.Equal(shardCoverageRoot.GetProperty("missingClassCount").GetInt32(), root.GetProperty("projectQualityShardCoverageMissingClassCount").GetInt32());
        Assert.Equal(shardCoverageRoot.GetProperty("invalidEvidenceCount").GetInt32(), root.GetProperty("projectQualityShardCoverageInvalidEvidenceCount").GetInt32());
        Assert.Equal("passed", root.GetProperty("projectQualityReleaseTimeoutSingleClassRunState").GetString());
        Assert.Equal("blocked-final-proof-readiness-owner-action-required", root.GetProperty("finalProofReadinessBlockerDashboardState").GetString());
        Assert.Equal(6, root.GetProperty("finalProofReadinessBlockerDashboardBlockerCount").GetInt32());
        Assert.Equal(6, root.GetProperty("finalProofReadinessBlockerDashboardOwnerActionRequiredCount").GetInt32());
        Assert.Equal(0, root.GetProperty("finalProofReadinessBlockerDashboardReadyForPromotionCount").GetInt32());
        Assert.False(root.GetProperty("finalProofReadinessBlockerDashboardCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("finalProofReadinessBlockerDashboardCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("finalProofReadinessBlockerDashboardCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("finalProofReadinessBlockerDashboardIsRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("finalProofReadinessBlockerDashboardIsPostPublishProof").GetBoolean());

        bool shardCoverageComplete =
            shardCoverageRoot.GetProperty("coverageState").GetString() == "complete-class-coverage" &&
            shardCoverageRoot.GetProperty("inventoryClassCount").GetInt32() == shardCoverageRoot.GetProperty("coveredClassCount").GetInt32() &&
            shardCoverageRoot.GetProperty("missingClassCount").GetInt32() == 0 &&
            shardCoverageRoot.GetProperty("invalidEvidenceCount").GetInt32() == 0;
        JsonElement[] items = root.GetProperty("evidenceItems").EnumerateArray().ToArray();
        AssertEvidenceItem(items, "yolovision-reference-asset-acquisition", passed: false, "license owner review remains required");
        AssertEvidenceItem(items, "yolovision-reference-asset-license-approval", passed: false, "owner-action evidence only");
        AssertEvidenceItem(items, "trt10-bridge-runtime-consumer", passed: true, "not public clean package-consumer proof");
        AssertEvidenceItem(items, "trt11-bridge-runtime-consumer", passed: false, "failed compatible-host attempt is evidence");
        AssertEvidenceItem(items, "trt11-runtime-smoke-root-cause-report", passed: false, "diagnostic blocker evidence only");
        AssertEvidenceItem(items, "trt11-runtime-dll-resolution-report", passed: false, "diagnostic blocker evidence only");
        AssertEvidenceItem(items, "trt10-vs-trt11-bridge-runtime-diagnostic-diff", passed: false, "diagnostic blocker evidence only");
        AssertEvidenceItem(items, "clean-public-package-consumer-proof-gap-report", passed: false, "owner-action planning evidence only");
        AssertEvidenceItem(items, "project-quality-shard-class-coverage", passed: shardCoverageComplete, "not a one-shot whole-suite pass");
        AssertEvidenceItem(items, "final-proof-readiness-blocker-dashboard", passed: false, "owner-action aggregation only");

        string[] sourceEvidence = root.GetProperty("sourceEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/yolovision/reference-assets/acquisition-report.json", sourceEvidence);
        Assert.Contains("artifacts/yolovision/reference-assets/asset-license-approval-validation.json", sourceEvidence);
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt10.11-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceEvidence);
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda13.2-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceEvidence);
        Assert.Contains("artifacts/final-release/trt11-runtime-smoke-root-cause-report.json", sourceEvidence);
        Assert.Contains("artifacts/final-release/trt11-runtime-dll-resolution-report.json", sourceEvidence);
        Assert.Contains("artifacts/final-release/trt10-vs-trt11-bridge-runtime-diagnostic-diff.json", sourceEvidence);
        Assert.Contains("artifacts/final-release/clean-public-package-consumer-proof-gap-report.json", sourceEvidence);
        Assert.Contains("artifacts/test-analysis/project-quality-shard-class-coverage.json", sourceEvidence);
        Assert.Contains("artifacts/test-analysis/project-quality-shards/20260711-release-timeout-single-class/summary.json", sourceEvidence);
        Assert.Contains("artifacts/final-release/final-proof-readiness-blocker-dashboard.json", sourceEvidence);

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-evidence-bundle.md"));
        Assert.Contains("YoloVision reference assets", markdown, StringComparison.Ordinal);
        Assert.Contains("YoloVision reference asset license approval", markdown, StringComparison.Ordinal);
        Assert.Contains("YoloVision asset license approval owner actions", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT10 bridge runtime consumer", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 bridge runtime consumer", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 root cause report", markdown, StringComparison.Ordinal);
        Assert.Contains("nativeCreateDiag=", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 DLL resolution report", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT10 vs TRT11 bridge diagnostic diff", markdown, StringComparison.Ordinal);
        Assert.Contains("Clean public package consumer proof gap report", markdown, StringComparison.Ordinal);
        Assert.Contains("createInferRuntime-null", markdown, StringComparison.Ordinal);
        Assert.Contains("ProjectQuality shard coverage", markdown, StringComparison.Ordinal);
        Assert.Contains("oneShotWholeSuite=False", markdown, StringComparison.Ordinal);
        Assert.Contains("final proof readiness blocker dashboard", markdown, StringComparison.Ordinal);
    }

    private static void AssertEvidenceItem(JsonElement[] items, string id, bool passed, string boundaryMarker)
    {
        JsonElement item = items.Single(item => item.GetProperty("id").GetString() == id);
        Assert.Equal(passed, item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryMarker, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
