using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerRollbackReviewAndCloseDecisionTests
{
    [Fact]
    public void RollbackReviewAndCloseDecisionImportsRemainBlockedWithoutOwnerProof()
    {
        RunPowerShell("Import-FinalOwnerRollbackReview.ps1");
        RunPowerShell("Test-FinalOwnerRollbackReview.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerCloseDecision.ps1");
        RunPowerShell("Test-FinalOwnerCloseDecision.ps1", "-Strict");

        using JsonDocument rollbackDocument = ReadFinalReleaseJson("final-owner-rollback-review-import.json");
        JsonElement rollback = rollbackDocument.RootElement;
        Assert.Equal("blocked-final-owner-rollback-review-required", rollback.GetProperty("importState").GetString());
        Assert.False(rollback.GetProperty("rollbackReviewReady").GetBoolean());
        Assert.True(rollback.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(rollback.GetProperty("failedActionRequiredCount").GetInt32() >= 15);
        Assert.True(rollback.GetProperty("rollbackPlanRequired").GetBoolean());
        Assert.True(rollback.GetProperty("rollbackOrWithdrawExecutionForbidden").GetBoolean());
        Assert.True(rollback.GetProperty("deleteDelistWithdrawDeprecateForbidden").GetBoolean());
        Assert.False(rollback.GetProperty("rollbackPlanIsReleaseCloseProof").GetBoolean());
        Assert.False(rollback.GetProperty("rollbackPlanIsPublicPublishProof").GetBoolean());
        AssertStringArrayContainsAll(
            rollback.GetProperty("forbiddenReleaseActions"),
            "delete",
            "delist",
            "withdraw",
            "deprecate",
            "dotnet nuget delete",
            "nuget delete");
        string rollbackBoundary = rollback.GetProperty("boundary").GetString()!;
        Assert.Contains("does not execute rollback", rollbackBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("delete", rollbackBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("withdraw", rollbackBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close proof", rollbackBoundary, StringComparison.OrdinalIgnoreCase);
        AssertNonProof(rollback);

        using JsonDocument rollbackValidationDocument = ReadFinalReleaseJson("final-owner-rollback-review-validation.json");
        JsonElement rollbackValidation = rollbackValidationDocument.RootElement;
        Assert.Equal("final-owner-rollback-review-validation-ready-non-proof", rollbackValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, rollbackValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(rollbackValidation.GetProperty("rollbackPlanRequired").GetBoolean());
        Assert.True(rollbackValidation.GetProperty("rollbackOrWithdrawExecutionForbidden").GetBoolean());
        Assert.True(rollbackValidation.GetProperty("deleteDelistWithdrawDeprecateForbidden").GetBoolean());
        Assert.False(rollbackValidation.GetProperty("rollbackPlanIsReleaseCloseProof").GetBoolean());
        AssertNonProof(rollbackValidation);

        using JsonDocument closeDocument = ReadFinalReleaseJson("final-owner-close-decision-import.json");
        JsonElement close = closeDocument.RootElement;
        Assert.Equal("blocked-final-owner-close-decision-required", close.GetProperty("importState").GetString());
        Assert.False(close.GetProperty("finalCloseDecisionReady").GetBoolean());
        Assert.True(close.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(close.GetProperty("failedActionRequiredCount").GetInt32() >= 8);
        Assert.False(close.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(close.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument closeValidationDocument = ReadFinalReleaseJson("final-owner-close-decision-validation.json");
        JsonElement closeValidation = closeValidationDocument.RootElement;
        Assert.Equal("final-owner-close-decision-validation-ready-non-proof", closeValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, closeValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProof(closeValidation);
    }

    [Fact]
    public void ReleaseEvidenceCarriesOwnerBackfillAndDecisionImportsAsFailedNonProof()
    {
        RunPowerShell("Export-OwnerRealProofEvidenceBackfillPackage.ps1");
        RunPowerShell("Test-OwnerRealProofEvidenceBackfillPackage.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealProofStagingWorkspaceContract.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspaceContract.ps1", "-Strict");
        RunPowerShell("Import-OwnerRealProofStagingWorkspace.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerRollbackReview.ps1");
        RunPowerShell("Test-FinalOwnerRollbackReview.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerCloseDecision.ps1");
        RunPowerShell("Test-FinalOwnerCloseDecision.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerRealProofConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        foreach (string id in new[]
        {
            "owner-real-proof-evidence-backfill-package",
            "owner-real-proof-staging-workspace-contract",
            "owner-real-proof-staging-workspace-import",
            "final-owner-rollback-review-import",
            "final-owner-close-decision-import"
        })
        {
            JsonElement item = bundle.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            string boundary = item.GetProperty("boundary").GetString()!;
            Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
    }

    private static void AssertNonProof(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
