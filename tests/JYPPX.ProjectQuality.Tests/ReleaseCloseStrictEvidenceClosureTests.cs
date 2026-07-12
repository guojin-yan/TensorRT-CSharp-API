using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseStrictEvidenceClosureTests
{
    [Fact]
    public void StrictEvidenceClosureAggregatesOwnerEvidenceAndBlocksFakeReadySubstitutes()
    {
        RunStrictClosurePipeline();

        using JsonDocument closureDocument = ReadFinalReleaseJson("release-close-strict-evidence-closure.json");
        JsonElement closure = closureDocument.RootElement;

        Assert.Equal("release-close-strict-evidence-closure", closure.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-evidence-closure-owner-evidence-required", closure.GetProperty("closureState").GetString());
        Assert.False(closure.GetProperty("strictEvidenceClosureReady").GetBoolean());
        Assert.Equal(8, closure.GetProperty("laneCount").GetInt32());
        Assert.True(closure.GetProperty("blockedLaneCount").GetInt32() >= 5);
        Assert.True(closure.GetProperty("failedCrossCheckCount").GetInt32() >= 1);
        Assert.Equal(10, closure.GetProperty("fakeReadySubstituteCaseCount").GetInt32());
        Assert.Equal(10, closure.GetProperty("blockedFakeReadySubstituteCaseCount").GetInt32());
        Assert.False(closure.GetProperty("performsPublish").GetBoolean());
        Assert.False(closure.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(closure.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(closure.GetProperty("failedBlockerCountIsNotProof").GetBoolean());
        Assert.False(closure.GetProperty("dashboardIsProof").GetBoolean());
        Assert.False(closure.GetProperty("dryRunIsProof").GetBoolean());
        Assert.False(closure.GetProperty("manualApprovalIsProof").GetBoolean());
        Assert.False(closure.GetProperty("queuedWorkflowIsProof").GetBoolean());
        Assert.False(closure.GetProperty("missingRunnerIsProof").GetBoolean());

        AssertStringArrayContainsAll(
            closure.GetProperty("forbiddenNonProofSubstitutes"),
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "dashboard",
            "dry-run",
            "manual approval",
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "sidecar-only",
            "TensorRtExec report");

        AssertLaneExists(closure, "owner-public-publish-result");
        AssertLaneExists(closure, "post-publish-owner-input");
        AssertLaneExists(closure, "post-publish-record");
        AssertLaneExists(closure, "rollback-review");
        AssertLaneExists(closure, "close-decision");
        AssertLaneExists(closure, "release-evidence-bundle");
        AssertLaneExists(closure, "classification-audit");
        AssertLaneExists(closure, "final-public-publish-acceptance-gate");

        foreach (JsonElement fakeReadyCase in closure.GetProperty("fakeReadySubstituteCases").EnumerateArray())
        {
            Assert.True(fakeReadyCase.GetProperty("fakeReadyShapeBlocked").GetBoolean());
            Assert.False(fakeReadyCase.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(fakeReadyCase.GetProperty("isReleaseCloseProof").GetBoolean());
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-strict-evidence-closure-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-close-strict-evidence-closure-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("release-close-strict-evidence-closure-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void StrictEvidenceClosureDashboardGroupsOwnerMaterialsAndRemainsNonProof()
    {
        RunStrictClosurePipeline();
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosureDashboard.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosureDashboard.ps1", "-Strict");

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("release-close-strict-evidence-closure-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;

        Assert.Equal("release-close-strict-evidence-closure-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-evidence-closure-dashboard-owner-action-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(6, dashboard.GetProperty("dashboardGroupCount").GetInt32());
        Assert.True(dashboard.GetProperty("blockedDashboardGroupCount").GetInt32() >= 1);
        Assert.False(dashboard.GetProperty("dashboardIsProof").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(dashboard.GetProperty("isReleaseCloseProof").GetBoolean());

        AssertDashboardGroupExists(dashboard, "publish-result");
        AssertDashboardGroupExists(dashboard, "post-publish-proof");
        AssertDashboardGroupExists(dashboard, "rollback-review");
        AssertDashboardGroupExists(dashboard, "close-decision");
        AssertDashboardGroupExists(dashboard, "classification-audit");
        AssertDashboardGroupExists(dashboard, "forbidden-substitute-scan");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-strict-evidence-closure-dashboard.md"));
        Assert.Contains("Required Artifact", markdown, StringComparison.Ordinal);
        Assert.Contains("Required Hash", markdown, StringComparison.Ordinal);
        Assert.Contains("Validator", markdown, StringComparison.Ordinal);
        Assert.Contains("dashboardIsProof", markdown, StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-strict-evidence-closure-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-close-strict-evidence-closure-dashboard-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("dashboardIsProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void RunStrictClosurePipeline()
    {
        RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishForbiddenSubstituteScan.ps1");
        RunPowerShell("Test-PublicPublishForbiddenSubstituteScan.ps1", "-Strict");
        RunPowerShell("Export-PostPublishVerificationOwnerInputTemplate.ps1");
        RunPowerShell("Test-PostPublishVerificationOwnerInput.ps1", "-Strict");
        RunPowerShell("Export-PostPublishVerificationRecordFromOwnerInput.ps1", "-OwnerInputPath", "artifacts/final-release/post-publish-verification-owner-input.template.json");
        RunPowerShell("Test-PostPublishVerificationRecord.ps1", "-InputPath", "artifacts/final-release/post-publish-verification-record.json");
        RunPowerShell("Import-FinalOwnerRollbackReview.ps1");
        RunPowerShell("Test-FinalOwnerRollbackReview.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerCloseDecision.ps1");
        RunPowerShell("Test-FinalOwnerCloseDecision.ps1", "-Strict");
        RunPowerShell("Test-FinalPublicPublishAcceptanceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosure.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosure.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertLaneExists(JsonElement closure, string id)
    {
        Assert.Contains(closure.GetProperty("lanes").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("artifact").GetString()!.StartsWith("artifacts/final-release/", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains(".ps1", StringComparison.Ordinal) &&
            !item.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void AssertDashboardGroupExists(JsonElement dashboard, string id)
    {
        Assert.Contains(dashboard.GetProperty("groups").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            !string.IsNullOrWhiteSpace(item.GetProperty("requiredArtifact").GetString()) &&
            !string.IsNullOrWhiteSpace(item.GetProperty("requiredHash").GetString()) &&
            !string.IsNullOrWhiteSpace(item.GetProperty("validatorCommand").GetString()) &&
            !item.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertStringArrayContainsAll(JsonElement array, params string[] expected)
    {
        HashSet<string> actual = array.EnumerateArray()
            .Select(item => item.GetString() ?? string.Empty)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (string item in expected)
        {
            Assert.Contains(item, actual);
        }
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
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
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName} {string.Join(' ', arguments)}{Environment.NewLine}{output}{Environment.NewLine}{error}");
        return output;
    }
}
