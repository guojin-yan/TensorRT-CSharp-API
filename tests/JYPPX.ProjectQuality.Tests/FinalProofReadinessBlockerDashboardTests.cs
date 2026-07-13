using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalProofReadinessBlockerDashboardTests
{
    [Fact]
    public void DashboardAggregatesFinalProofBlockersWithoutPromotion()
    {
        string output = RunPowerShell("Export-FinalProofReadinessBlockerDashboard.ps1");
        Assert.Contains("Final proof readiness blocker dashboard written", output, StringComparison.Ordinal);

        using JsonDocument document = ReadFinalReleaseJson("final-proof-readiness-blocker-dashboard.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-proof-readiness-blocker-dashboard", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-proof-readiness-owner-action-required", root.GetProperty("dashboardState").GetString());
        Assert.Equal(6, root.GetProperty("blockerCount").GetInt32());
        Assert.Equal(6, root.GetProperty("ownerActionRequiredCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyForPromotionCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement sourceStates = root.GetProperty("sourceStates");
        Assert.Equal("blocked-evidence-incomplete", sourceStates.GetProperty("releaseEvidenceBundleState").GetString());
        Assert.Equal("blocked-final-release-close-owner-action-required", sourceStates.GetProperty("closeDashboardState").GetString());
        Assert.Equal("blocked-clean-public-package-consumer-proof-owner-action-required", sourceStates.GetProperty("cleanPublicPackageConsumerProofGapReportState").GetString());
        Assert.Equal(6, sourceStates.GetProperty("cleanPublicPackageConsumerProofGapCount").GetInt32());
        Assert.Equal(6, sourceStates.GetProperty("cleanPublicPackageConsumerProofOwnerActionRequiredCount").GetInt32());
        Assert.Equal(31, sourceStates.GetProperty("cleanPublicPackageConsumerProofOwnerInputFailedActionRequiredCount").GetInt32());
        Assert.False(sourceStates.GetProperty("cleanPublicPackageConsumerProofCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-required", sourceStates.GetProperty("postPublishPreflightState").GetString());
        Assert.Equal("compatible-host-bridge-package-runtime-failed", sourceStates.GetProperty("trt11ProofClassification").GetString());
        Assert.Equal("failed", sourceStates.GetProperty("trt11SmokeStatus").GetString());
        Assert.Equal("classified-owner-action-required", sourceStates.GetProperty("trt11RootCauseReportState").GetString());
        Assert.Equal("createInferRuntime-null", sourceStates.GetProperty("trt11RootCauseFailureSignature").GetString());
        Assert.Equal("trt11-create-runtime-null-cuda-runtime-error", sourceStates.GetProperty("trt11RootCauseCategory").GetString());
        Assert.True(sourceStates.TryGetProperty("trt11RootCauseNativeCreateRuntimeDiagnosticAvailable", out _));
        Assert.True(sourceStates.TryGetProperty("trt11RootCauseNativeCreateRuntimeAttempted", out _));
        Assert.True(sourceStates.TryGetProperty("trt11RootCauseNativeCreateRuntimeReturnedNull", out _));
        Assert.True(sourceStates.TryGetProperty("trt11RootCauseNativeCreateRuntimeLastStatus", out _));
        Assert.Equal("after-createInferRuntime-null-guard-ok", sourceStates.GetProperty("trt11RootCauseNativeCreateRuntimePhase").GetString());
        Assert.True(int.Parse(sourceStates.GetProperty("trt11RootCauseNativeCreateRuntimeLoggerMessageCount").GetString()!) >= 1);
        Assert.Matches("Cuda Runtime|catchCudaError", sourceStates.GetProperty("trt11RootCauseNativeCreateRuntimeLastLoggerMessage").GetString()!);
        Assert.False(sourceStates.GetProperty("trt11RootCauseCanPromoteRuntimeProof").GetBoolean());
        Assert.StartsWith("diagnostic-", sourceStates.GetProperty("trt11DllResolutionReportState").GetString(), StringComparison.Ordinal);
        Assert.True(sourceStates.GetProperty("trt11DllResolutionMissingRequiredDllGroupCount").GetInt32() >= 0);
        Assert.True(sourceStates.GetProperty("trt11DllResolutionDuplicateDllGroupCount").GetInt32() >= 0);
        Assert.False(sourceStates.GetProperty("trt11DllResolutionCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("diagnostic-diff-ready-non-proof", sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffState").GetString());
        Assert.True(sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffDifferingFieldCount").GetInt32() >= 10);
        Assert.True(sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt10SmokePassed").GetBoolean());
        Assert.True(sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt11SmokeFailed").GetBoolean());
        Assert.Equal("createInferRuntime-null", sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffFailureSignature").GetString());
        Assert.False(sourceStates.GetProperty("trt10VsTrt11BridgeRuntimeDiagnosticDiffCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("owner-action-required", sourceStates.GetProperty("yoloVisionLicenseApprovalState").GetString());
        Assert.Equal(22, sourceStates.GetProperty("yoloVisionLicenseOwnerActionRequiredCount").GetInt32());
        Assert.Equal("ready-for-next-implementation-batch", sourceStates.GetProperty("deferredBTierWorkPackageState").GetString());
        Assert.Equal("runbook-ready-non-proof", sourceStates.GetProperty("projectQualityShardRunbookState").GetString());
        Assert.True(sourceStates.GetProperty("projectQualityShardRunbookReady").GetBoolean());

        string[] blockerIds = root.GetProperty("blockers").EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string required in new[]
        {
            "clean-public-package-consumer-proof",
            "post-publish-clean-consumer-proof",
            "yolovision-asset-license-approval",
            "trt11-runtime-smoke-root-cause",
            "deferred-readonly-implementation-batch",
            "project-quality-shard-runbook"
        })
        {
            Assert.Contains(required, blockerIds);
        }

        JsonElement cleanLane = root.GetProperty("blockers").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "clean-public-package-consumer-proof");
        Assert.Contains("blocked-clean-public-package-consumer-proof-owner-action-required", cleanLane.GetProperty("currentState").GetString(), StringComparison.Ordinal);
        Assert.Contains("gaps=6", cleanLane.GetProperty("currentState").GetString(), StringComparison.Ordinal);
        Assert.Contains(
            "clean public package consumer proof gap report",
            cleanLane.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!));
        string[] cleanRejects = cleanLane.GetProperty("rejects").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", cleanRejects);
        Assert.Contains("ProjectReference", cleanRejects);
        Assert.Contains("direct .nupkg", cleanRejects);
        Assert.Contains("bridge-only compatible-host smoke", cleanRejects);

        JsonElement postPublishLane = root.GetProperty("blockers").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "post-publish-clean-consumer-proof");
        Assert.Contains(
            "pre-publish smoke reused as post-publish proof",
            postPublishLane.GetProperty("rejects").EnumerateArray().Select(static item => item.GetString()!));

        JsonElement trt11Lane = root.GetProperty("blockers").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "trt11-runtime-smoke-root-cause");
        Assert.Contains("createInferRuntime-null", trt11Lane.GetProperty("currentState").GetString(), StringComparison.Ordinal);
        Assert.Contains("createDiag=", trt11Lane.GetProperty("currentState").GetString(), StringComparison.Ordinal);
        Assert.Contains("diagnostic-diff-ready-non-proof", trt11Lane.GetProperty("currentState").GetString(), StringComparison.Ordinal);
        Assert.Contains(
            "TRT11 runtime smoke root-cause report",
            trt11Lane.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(
            "TRT11 runtime DLL resolution report",
            trt11Lane.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(
            "TRT10/TRT11 bridge runtime diagnostic diff",
            trt11Lane.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(
            "TRT11 native create-runtime diagnostic snapshot",
            trt11Lane.GetProperty("requiredEvidence").EnumerateArray().Select(static item => item.GetString()!));

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/clean-public-package-consumer-proof-gap-report.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-post-publish-clean-consumer-proof-preflight.json", sourceArtifacts);
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda13.2-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt11-runtime-smoke-root-cause-report.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt11-runtime-dll-resolution-report.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt10-vs-trt11-bridge-runtime-diagnostic-diff.json", sourceArtifacts);
        Assert.Contains("artifacts/interface-coverage/deferred-btier-implementation-work-package.json", sourceArtifacts);
        Assert.Contains("artifacts/test-analysis/project-quality-shard-runbook.md", sourceArtifacts);
        Assert.Contains("docs/articles/zh-cn/project-quality-sharded-gate.md", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-proof-readiness-blocker-dashboard.md"));
        Assert.Contains("Final Proof Readiness Blocker Dashboard", markdown, StringComparison.Ordinal);
        Assert.Contains("clean-public-package-consumer-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Clean public package consumer proof gap report", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 runtime smoke", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 root cause report", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT11 DLL resolution report", markdown, StringComparison.Ordinal);
        Assert.Contains("TRT10 vs TRT11 bridge diagnostic diff", markdown, StringComparison.Ordinal);
        Assert.Contains("createInferRuntime-null", markdown, StringComparison.Ordinal);
        Assert.Contains("YoloVision asset license approval", markdown, StringComparison.Ordinal);
        Assert.Contains("ProjectQuality shard runbook", markdown, StringComparison.Ordinal);
        Assert.Contains("runbook-ready-non-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("does not run runtime smoke", markdown, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string RunPowerShell(string scriptName)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
