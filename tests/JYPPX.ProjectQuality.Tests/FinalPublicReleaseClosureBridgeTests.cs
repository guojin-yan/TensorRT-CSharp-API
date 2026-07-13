using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalPublicReleaseClosureBridgeTests
{
    [Fact]
    public void FinalPublicReleaseClosureBridgeKeepsRealOwnerProofBlockedAndSideEffectFree()
    {
        RunPowerShell("Export-PublicPackageDownloadProofInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPackageDownloadProofCandidate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofCandidate.ps1", "-Strict");
        RunPowerShell("Export-CleanExternalConsumerSmokeInputTemplate.ps1");
        RunPowerShell("Test-CleanExternalConsumerSmokeInput.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublishAuthorizationInputTemplate.ps1");
        RunPowerShell("Test-OwnerPublishAuthorizationInput.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublishExecutionResultInputTemplate.ps1");
        RunPowerShell("Test-OwnerPublishExecutionResultInput.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerProofResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");
        RunPowerShell("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseReadyConvergenceDashboard.ps1");
        RunPowerShell("Test-StrictCloseReadyConvergenceDashboard.ps1", "-Strict");
        RunPowerShell("Export-FinalPublicReleaseClosureBridge.ps1");
        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-Strict");

        using JsonDocument bridgeDocument = ReadFinalReleaseJson("final-public-release-closure-bridge.json");
        JsonElement bridge = bridgeDocument.RootElement;
        Assert.Equal("final-public-release-closure-bridge", bridge.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-public-release-closure-real-owner-proof-required", bridge.GetProperty("bridgeState").GetString());
        Assert.Equal(7, bridge.GetProperty("laneCount").GetInt32());
        Assert.True(bridge.GetProperty("blockedLaneCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(bridge);

        string[] laneIds = bridge.GetProperty("closureLanes").EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        Assert.Contains("owner-publish-authorization", laneIds);
        Assert.Contains("owner-publish-execution-result", laneIds);
        Assert.Contains("public-package-download-proof", laneIds);
        Assert.Contains("clean-external-consumer-smoke", laneIds);
        Assert.Contains("post-publish-proof", laneIds);
        Assert.Contains("release-issue-close-owner-decision", laneIds);
        Assert.Contains("strict-close-ready-convergence-dashboard", laneIds);

        foreach (JsonElement lane in bridge.GetProperty("closureLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("performsPublish").GetBoolean());
            Assert.False(lane.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(lane.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerAction").GetString()));
            AssertHasNonProofBoundary(lane.GetProperty("boundary").GetString());
        }

        string[] sourceArtifacts = bridge.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-publish-authorization-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-publish-execution-result-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-package-download-proof-candidate-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/clean-external-consumer-smoke-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json", sourceArtifacts);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-public-release-closure-bridge-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-public-release-closure-bridge-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-public-release-closure-real-owner-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.Equal(7, validation.GetProperty("laneCount").GetInt32());
        AssertFalseProofPublishCloseFlags(validation);
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
        Assert.False(element.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertHasNonProofBoundary(string? boundary)
    {
        Assert.False(string.IsNullOrWhiteSpace(boundary));
        Assert.True(
            boundary.Contains("not", StringComparison.OrdinalIgnoreCase) ||
            boundary.Contains("never", StringComparison.OrdinalIgnoreCase) ||
            boundary.Contains("cannot", StringComparison.OrdinalIgnoreCase),
            $"Boundary must explicitly reject promotion or side effects: {boundary}");
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(
            process.ExitCode == 0,
            $"PowerShell script failed: {scriptName} {string.Join(' ', arguments)}{Environment.NewLine}STDOUT:{Environment.NewLine}{stdout}{Environment.NewLine}STDERR:{Environment.NewLine}{stderr}");

        return stdout;
    }
}
