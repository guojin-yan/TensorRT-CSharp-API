using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalReadonlyPublishAuditAndOwnerManualTests
{
    [Fact]
    public void FinalReadonlyPublishAuditPackAuditsWorkflowsAndStaysNonProof()
    {
        RunReadonlyAuditPipeline();

        using JsonDocument document = ReadFinalReleaseJson("final-readonly-publish-audit-pack.json");
        JsonElement pack = document.RootElement;
        Assert.Equal("final-readonly-publish-audit-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-readonly-publish-audit-owner-action-required", pack.GetProperty("auditState").GetString());
        Assert.Equal(12, pack.GetProperty("auditLaneCount").GetInt32());
        Assert.True(pack.GetProperty("workflowAuditCount").GetInt32() >= 1);
        Assert.True(pack.GetProperty("packageWorkflowCount").GetInt32() >= 1);
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsGitHubPackagesPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsNuGetPublish").GetBoolean());
        Assert.False(pack.GetProperty("executesDeleteDelistWithdrawDeprecate").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("queuedWorkflowIsProof").GetBoolean());
        Assert.False(pack.GetProperty("missingSelfHostedRunnerIsProof").GetBoolean());
        Assert.False(pack.GetProperty("manualApprovalIsProof").GetBoolean());
        Assert.False(pack.GetProperty("dashboardIsProof").GetBoolean());
        Assert.False(pack.GetProperty("dryRunIsProof").GetBoolean());
        AssertStringArrayContainsAll(pack.GetProperty("forbiddenNonProofSubstitutes"), RequiredSubstitutes);
        AssertLaneExists(pack, "strict-closure");
        AssertLaneExists(pack, "strict-closure-dashboard");
        AssertLaneExists(pack, "final-acceptance-gate");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-readonly-publish-audit-pack.md"));
        Assert.Contains("GitHub Workflow Readonly Audit", markdown, StringComparison.Ordinal);
        Assert.Contains("Required", markdown, StringComparison.Ordinal);
        Assert.Contains("queued GitHub Actions run", markdown, StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-readonly-publish-audit-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-readonly-publish-audit-pack-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void FinalOwnerOneScreenManualCapturesExecutionOrderWithoutExecutingPublish()
    {
        RunReadonlyAuditPipeline();
        RunPowerShell("Export-FinalOwnerOneScreenExecutionManual.ps1");
        RunPowerShell("Test-FinalOwnerOneScreenExecutionManual.ps1", "-Strict");

        using JsonDocument document = ReadFinalReleaseJson("final-owner-one-screen-execution-manual.json");
        JsonElement manual = document.RootElement;
        Assert.Equal("final-owner-one-screen-execution-manual", manual.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-one-screen-execution-manual-owner-action-required", manual.GetProperty("manualState").GetString());
        Assert.Equal(8, manual.GetProperty("stepCount").GetInt32());
        Assert.True(manual.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(manual.GetProperty("manualIsProof").GetBoolean());
        Assert.False(manual.GetProperty("performsPublish").GetBoolean());
        Assert.False(manual.GetProperty("performsGitHubPackagesPublish").GetBoolean());
        Assert.False(manual.GetProperty("performsNuGetPublish").GetBoolean());
        Assert.False(manual.GetProperty("canCloseReleaseIssue").GetBoolean());
        AssertStringArrayContainsAll(manual.GetProperty("forbiddenNonProofSubstitutes"), ManualRequiredSubstitutes);
        Assert.Equal(8, manual.GetProperty("releaseCloseRealInputChainCount").GetInt32());
        Assert.True(manual.GetProperty("releaseCloseRealInputChainRequiredFieldCount").GetInt32() >= 100);
        Assert.True(manual.GetProperty("releaseCloseRealInputChainRejectedSubstituteCount").GetInt32() >= 30);
        Assert.Equal(18, manual.GetProperty("releaseCloseRealInputChainSourceReadinessSignalCount").GetInt32());
        Assert.True(manual.GetProperty("releaseCloseRealInputChainBlockedRealInputCount").GetInt32() > 0);
        Assert.True(manual.GetProperty("publicPackageDownloadProofRequiredFieldCount").GetInt32() >= 30);
        Assert.Equal(11, manual.GetProperty("publicPackageDownloadProofRejectedSubstituteCount").GetInt32());
        Assert.Equal(7, manual.GetProperty("publicPackageDownloadProofSourceReadinessSignalCount").GetInt32());
        Assert.False(manual.GetProperty("publicPackageDownloadProofCandidateReady").GetBoolean());
        Assert.True(manual.GetProperty("postPublishCleanConsumerProofRequiredFieldCount").GetInt32() >= 50);
        Assert.Equal(11, manual.GetProperty("postPublishCleanConsumerProofRejectedSubstituteCount").GetInt32());
        Assert.True(manual.GetProperty("postPublishCleanConsumerProofBlockedRealInputCount").GetInt32() > 0);
        Assert.Equal(11, manual.GetProperty("postPublishCleanConsumerProofSourceReadinessSignalCount").GetInt32());
        Assert.False(manual.GetProperty("postPublishCleanConsumerProofCandidateReady").GetBoolean());
        Assert.False(manual.GetProperty("postPublishCleanConsumerProofSourceProofLinkageReady").GetBoolean());
        Assert.Matches("^[0-9a-f]{64}$", manual.GetProperty("releaseEvidenceBundleSha256").GetString()!);
        Assert.Equal("blocked-final-close-gate-owner-proof-required", manual.GetProperty("finalCloseStrictValidatorOutputState").GetString());
        AssertIds(manual, "releaseCloseRealInputChain",
        [
            "github-actions-run-evidence",
            "owner-public-publish-result",
            "public-package-download-proof",
            "post-publish-clean-consumer-proof-result",
            "rollback-review",
            "final-close-decision",
            "release-evidence-bundle-sha",
            "strict-close-validator-output"
        ]);
        AssertStepExists(manual, "preflight-freeze");
        AssertStepExists(manual, "owner-authorization");
        AssertStepExists(manual, "public-publish-command");
        AssertStepExists(manual, "package-page-and-download");
        AssertStepExists(manual, "post-publish-clean-consumer");
        AssertStepExists(manual, "rollback-review");
        AssertStepExists(manual, "close-decision");
        AssertStepExists(manual, "strict-final-verification");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-one-screen-execution-manual.md"));
        Assert.Contains("Final Owner One-Screen Execution Manual", markdown, StringComparison.Ordinal);
        Assert.Contains("dotnet nuget push", manual.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("GitHub Packages", manual.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-one-screen-execution-manual-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-one-screen-execution-manual-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("releaseCloseRealInputChainCount").GetInt32());
        Assert.True(validation.GetProperty("releaseCloseRealInputChainRequiredFieldCount").GetInt32() >= 100);
        Assert.True(validation.GetProperty("releaseCloseRealInputChainRejectedSubstituteCount").GetInt32() >= 30);
        Assert.Equal(18, validation.GetProperty("releaseCloseRealInputChainSourceReadinessSignalCount").GetInt32());
        Assert.True(validation.GetProperty("releaseCloseRealInputChainBlockedRealInputCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("publicPackageDownloadProofCandidateReady").GetBoolean());
        Assert.False(validation.GetProperty("postPublishCleanConsumerProofCandidateReady").GetBoolean());
        Assert.False(validation.GetProperty("postPublishCleanConsumerProofSourceProofLinkageReady").GetBoolean());
        Assert.Matches("^[0-9a-f]{64}$", validation.GetProperty("releaseEvidenceBundleSha256").GetString()!);
        Assert.Equal("blocked-final-close-gate-owner-proof-required", validation.GetProperty("finalCloseStrictValidatorOutputState").GetString());
        Assert.False(validation.GetProperty("manualIsProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static readonly string[] RequiredSubstitutes =
    [
        "local feed",
        "ProjectReference",
        "direct .nupkg",
        "dashboard",
        "dry-run",
        "manual approval",
        "queued GitHub Actions run",
        "missing self-hosted runner",
        "sidecar-only",
        "TensorRtExec report",
    ];

    private static readonly string[] ManualRequiredSubstitutes =
    [
        .. RequiredSubstitutes,
        "public package download proof alone",
        "post-publish validation-ready without proofCandidateReady",
        "release evidence bundle hash only",
        "strict close validator output without real proof",
    ];

    private static void RunReadonlyAuditPipeline()
    {
        RunPowerShell("Export-FinalPublicPublishPreExecutionFreeze.ps1");
        RunPowerShell("Test-FinalPublicPublishPreExecutionFreeze.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultPreflight.ps1", "-Strict");
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
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosure.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosure.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosureDashboard.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosureDashboard.ps1", "-Strict");
        RunPowerShell("Test-FinalPublicPublishAcceptanceGate.ps1", "-Strict");
        RunPowerShell("Export-FinalReadonlyPublishAuditPack.ps1");
        RunPowerShell("Test-FinalReadonlyPublishAuditPack.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertLaneExists(JsonElement pack, string id)
    {
        Assert.Contains(pack.GetProperty("lanes").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            !item.GetProperty("performsPublish").GetBoolean() &&
            !item.GetProperty("canCloseReleaseIssue").GetBoolean() &&
            item.GetProperty("validatorCommand").GetString()!.Contains(".ps1", StringComparison.Ordinal));
    }

    private static void AssertStepExists(JsonElement manual, string id)
    {
        Assert.Contains(manual.GetProperty("steps").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("notExecutedByAutomation").GetBoolean() &&
            !item.GetProperty("performsPublish").GetBoolean() &&
            !item.GetProperty("canCloseReleaseIssue").GetBoolean() &&
            item.GetProperty("validatorCommand").GetString()!.Contains(".ps1", StringComparison.Ordinal));
    }

    private static void AssertIds(JsonElement root, string propertyName, string[] expectedIds)
    {
        string[] actual = root.GetProperty(propertyName)
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string expectedId in expectedIds)
        {
            Assert.Contains(expectedId, actual);
        }
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
