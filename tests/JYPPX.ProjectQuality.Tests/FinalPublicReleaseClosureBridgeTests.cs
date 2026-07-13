using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalPublicReleaseClosureBridgeTests
{
    private static readonly JsonSerializerOptions IndentedJsonOptions = new()
    {
        WriteIndented = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
    };

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
        Assert.Equal(9, bridge.GetProperty("laneCount").GetInt32());
        Assert.True(bridge.GetProperty("blockedLaneCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(bridge);
        Assert.Equal(0, bridge.GetProperty("failedConsistencyBlockerCount").GetInt32());
        Assert.True(bridge.GetProperty("failedConsistencyActionRequiredCount").GetInt32() > 0);
        Assert.Equal(0, bridge.GetProperty("forbiddenSubstituteFindingCount").GetInt32());

        string[] laneIds = bridge.GetProperty("closureLanes").EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        Assert.Contains("github-actions-run-proof", laneIds);
        Assert.Contains("owner-public-publish-result", laneIds);
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

        string[] checkIds = bridge.GetProperty("crossLaneConsistencyChecks").EnumerateArray()
            .Select(static check => check.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("github-actions-run-evidence-ready", checkIds);
        Assert.Contains("owner-public-publish-result-ready", checkIds);
        Assert.Contains("public-download-proof-ready", checkIds);
        Assert.Contains("forbidden-substitutes-absent", checkIds);

        JsonElement proofSummary = bridge.GetProperty("closureProofSourceSummary");
        Assert.False(proofSummary.GetProperty("githubActionsRunEvidenceReady").GetBoolean());
        Assert.False(proofSummary.GetProperty("ownerPublicPublishResultReady").GetBoolean());
        Assert.False(proofSummary.GetProperty("publicDownloadProofReady").GetBoolean());

        string[] sourceArtifacts = bridge.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/github-actions-run-evidence-import-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json", sourceArtifacts);
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
        Assert.Equal(9, validation.GetProperty("laneCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedConsistencyBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedConsistencyActionRequiredCount").GetInt32() > 0);
        Assert.Equal(0, validation.GetProperty("forbiddenSubstituteFindingCount").GetInt32());
        AssertFalseProofPublishCloseFlags(validation);
    }

    [Fact]
    public void FinalPublicReleaseClosureBridgeValidatorRejectsForbiddenSubstituteConsistencyBlocker()
    {
        RunPowerShell("Export-FinalPublicReleaseClosureBridge.ps1");

        string sourcePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-public-release-closure-bridge.json");
        JsonObject bridge = JsonNode.Parse(File.ReadAllText(sourcePath))!.AsObject();
        bridge["bridgeState"] = "invalid-final-public-release-closure-bridge";
        bridge["failedConsistencyBlockerCount"] = 1;
        bridge["forbiddenSubstituteFindingCount"] = 1;

        JsonArray checks = bridge["crossLaneConsistencyChecks"]!.AsArray();
        JsonObject forbiddenCheck = checks.Single(static check => check!["id"]!.GetValue<string>() == "forbidden-substitutes-absent")!.AsObject();
        forbiddenCheck["passed"] = false;
        forbiddenCheck["severity"] = "blocker";
        forbiddenCheck["detail"] = "Forbidden substitute findings were propagated: local feed";

        string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-public-release-closure-bridge.misuse.json");
        File.WriteAllText(misusePath, bridge.ToJsonString(IndentedJsonOptions));

        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-InputPath", "artifacts/final-release/final-public-release-closure-bridge.misuse.json");

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-public-release-closure-bridge-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("invalid-final-public-release-closure-bridge", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
        Assert.Equal(1, validation.GetProperty("failedConsistencyBlockerCount").GetInt32());
    }

    [Fact]
    public void FinalPublicReleaseClosureBridgeValidatorAcceptsReadyShapedAllLaneFixtureWithoutSideEffects()
    {
        string readyPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-public-release-closure-bridge.ready.json");
        File.WriteAllText(readyPath, BuildReadyBridgeFixture().ToJsonString(IndentedJsonOptions));

        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-InputPath", "artifacts/final-release/final-public-release-closure-bridge.ready.json", "-Strict");

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-public-release-closure-bridge-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-public-release-closure-bridge-ready-for-owner-close-review", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedConsistencyBlockerCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedConsistencyActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(validation);
    }

    private static JsonObject BuildReadyBridgeFixture()
    {
        string[] laneIds =
        [
            "github-actions-run-proof",
            "owner-public-publish-result",
            "owner-publish-authorization",
            "owner-publish-execution-result",
            "public-package-download-proof",
            "clean-external-consumer-smoke",
            "post-publish-proof",
            "release-issue-close-owner-decision",
            "strict-close-ready-convergence-dashboard",
        ];

        string[] sourceArtifacts =
        [
            "artifacts/final-release/github-actions-run-evidence-import-validation.json",
            "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
            "artifacts/final-release/owner-publish-authorization-input-validation.json",
            "artifacts/final-release/owner-publish-execution-result-input-validation.json",
            "artifacts/final-release/public-package-download-proof-candidate-validation.json",
            "artifacts/final-release/clean-external-consumer-smoke-input-validation.json",
            "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
            "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
            "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json",
        ];

        JsonArray lanes = [];
        foreach (string laneId in laneIds)
        {
            lanes.Add(new JsonObject
            {
                ["laneId"] = laneId,
                ["state"] = "ready",
                ["ready"] = true,
                ["blocked"] = false,
                ["performsPublish"] = false,
                ["usesPublishToken"] = false,
                ["canPublishPublicly"] = false,
                ["canCloseReleaseIssue"] = false,
                ["isReleaseCloseProof"] = false,
                ["ownerAction"] = "Ready-shaped fixture action only.",
                ["boundary"] = "Fixture cannot publish, cannot use tokens, cannot promote proof, and cannot close a release issue.",
            });
        }

        JsonArray checks = [];
        foreach (string checkId in RequiredConsistencyCheckIds())
        {
            checks.Add(new JsonObject
            {
                ["id"] = checkId,
                ["passed"] = true,
                ["severity"] = checkId == "forbidden-substitutes-absent" ? "blocker" : "action-required",
                ["detail"] = "Ready-shaped fixture check passed.",
            });
        }

        JsonArray artifactNodes = [];
        foreach (string artifact in sourceArtifacts)
        {
            artifactNodes.Add(artifact);
        }

        return new JsonObject
        {
            ["recordKind"] = "final-public-release-closure-bridge",
            ["generatedAtUtc"] = DateTimeOffset.UtcNow.ToString("O"),
            ["bridgeState"] = "final-public-release-closure-bridge-ready-for-owner-close-review",
            ["laneCount"] = laneIds.Length,
            ["readyLaneCount"] = laneIds.Length,
            ["blockedLaneCount"] = 0,
            ["missingArtifactCount"] = 0,
            ["failedConsistencyBlockerCount"] = 0,
            ["failedConsistencyActionRequiredCount"] = 0,
            ["forbiddenSubstituteFindingCount"] = 0,
            ["notExecutedByAutomation"] = true,
            ["performsPublish"] = false,
            ["usesPublishToken"] = false,
            ["canPromoteRuntimeProof"] = false,
            ["canPublishPublicly"] = false,
            ["canCloseReleaseIssue"] = false,
            ["isRuntimeExecutionProof"] = false,
            ["isPackageConsumerRuntimeProof"] = false,
            ["isPostPublishProof"] = false,
            ["isReleaseCloseProof"] = false,
            ["allLanesSideEffectSafe"] = true,
            ["closureLanes"] = lanes,
            ["crossLaneConsistencyChecks"] = checks,
            ["closureProofSourceSummary"] = new JsonObject
            {
                ["githubActionsRunEvidenceReady"] = true,
                ["ownerPublicPublishResultReady"] = true,
                ["publicDownloadProofReady"] = true,
            },
            ["sourceArtifacts"] = artifactNodes,
            ["nextOwnerActions"] = new JsonArray(),
            ["safetyBoundary"] = "Ready-shaped validation fixture only; not runtime proof, not package push, and cannot close a release issue.",
        };
    }

    private static string[] RequiredConsistencyCheckIds()
    {
        return
        [
            "github-actions-run-evidence-ready",
            "github-actions-run-url-present",
            "github-actions-head-sha-format",
            "github-actions-log-and-artifact-hashes",
            "owner-public-publish-result-ready",
            "owner-public-publish-links-github-actions",
            "public-download-proof-ready",
            "public-download-links-source-proofs",
            "owner-and-public-download-package-url-match",
            "owner-and-public-download-version-match",
            "owner-and-public-download-sha-match",
            "runtime-package-url-public",
            "github-release-asset-consistent",
            "owner-reviewer-and-timestamp-present",
            "forbidden-substitutes-absent",
        ];
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
