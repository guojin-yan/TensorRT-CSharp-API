using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalPublicPublishAcceptanceGateTests
{
    [Fact]
    public void PreExecutionFreezeBlocksReleaseCloseAndRollbackExecution()
    {
        RunPowerShell("Export-FinalPublicPublishPreExecutionFreeze.ps1");
        string output = RunPowerShell("Test-FinalPublicPublishPreExecutionFreeze.ps1", "-Strict");

        Assert.Contains("ValidationState=blocked-public-publish-owner-action-required", output, StringComparison.Ordinal);

        using JsonDocument freezeDocument = ReadFinalReleaseJson("final-public-publish-pre-execution-freeze.json");
        JsonElement freeze = freezeDocument.RootElement;
        Assert.Equal("final-public-publish-pre-execution-freeze", freeze.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-owner-action-required", freeze.GetProperty("freezeState").GetString());
        Assert.False(freeze.GetProperty("performsPublish").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(freeze.GetProperty("releaseCloseCandidate").GetBoolean());
        Assert.True(freeze.GetProperty("rollbackPlanRequired").GetBoolean());
        Assert.True(freeze.GetProperty("rollbackOrWithdrawExecutionForbidden").GetBoolean());
        Assert.True(freeze.GetProperty("deleteDelistWithdrawDeprecateForbidden").GetBoolean());
        AssertStringArrayContainsAll(
            freeze.GetProperty("forbiddenReleaseActions"),
            "delete",
            "delist",
            "withdraw",
            "deprecate",
            "dotnet nuget delete",
            "nuget delete");
        AssertStringArrayContainsAll(
            freeze.GetProperty("nonProofSubstitutes"),
            "queued GitHub Actions run",
            "missing self-hosted runner",
            "manual approval",
            "dashboard",
            "dry-run",
            "local feed",
            "ProjectReference",
            "direct .nupkg");
        AssertStringArrayContainsAll(
            freeze.GetProperty("strictProofRequirements"),
            "owner authorization",
            "public publish result",
            "post-publish proof",
            "clean consumer runtime proof",
            "release close owner decision");
        string boundary = freeze.GetProperty("boundary").GetString()!;
        Assert.Contains("does not delete", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("withdraw", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("deprecate", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("strict owner authorization", boundary, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-public-publish-pre-execution-freeze-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-public-publish-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("releaseCloseCandidate").GetBoolean());
        Assert.True(validation.GetProperty("rollbackOrWithdrawExecutionForbidden").GetBoolean());
        Assert.True(validation.GetProperty("deleteDelistWithdrawDeprecateForbidden").GetBoolean());
    }

    [Fact]
    public void AcceptanceGateBlocksUntilRealOwnerPublishPostPublishAndCloseEvidenceExist()
    {
        RunPowerShell("Export-FinalPublicPublishCommandDryContract.ps1");
        RunPowerShell("Test-FinalPublicPublishCommandDryContract.ps1", "-Strict");
        RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerRealProofConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        string output = RunPowerShell("Test-FinalPublicPublishAcceptanceGate.ps1", "-Strict");

        Assert.Contains("FinalPublicPublishAcceptanceGateState=blocked-final-public-publish-owner-evidence-required", output, StringComparison.Ordinal);

        using JsonDocument gateDocument = ReadFinalReleaseJson("final-public-publish-acceptance-gate.json");
        JsonElement gate = gateDocument.RootElement;

        Assert.Equal("final-public-publish-acceptance-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-public-publish-owner-evidence-required", gate.GetProperty("gateState").GetString());
        Assert.True(gate.GetProperty("blockedGateCheckCount").GetInt32() >= 4);
        Assert.False(gate.GetProperty("readyForPublicReleaseClose").GetBoolean());
        Assert.True(gate.GetProperty("failedBlockerCountIsNotProof").GetBoolean());
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());
        Assert.False(gate.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(gate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(gate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("never executes dotnet nuget push", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("pre-publish smoke", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-public-publish-acceptance-gate.md"));
        Assert.DoesNotContain("System.Object[]", markdown, StringComparison.Ordinal);
        Assert.Contains("| ID | Passed | Source Artifact | Required Evidence | Blocking Reason | Owner Action Required |", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-public-publish-real-result-accepted", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-real-proof-accepted", markdown, StringComparison.Ordinal);
        Assert.Contains("final-release-close-approval-real-input-accepted", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-proof-convergence-ready", markdown, StringComparison.Ordinal);

        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-public-publish-real-result-accepted" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-real-proof-accepted" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-release-close-approval-real-input-accepted" &&
            !item.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndClassificationAuditCarryAcceptanceGateAsNonProof()
    {
        RunPowerShell("Test-FinalPublicPublishAcceptanceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Contains(bundle.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-public-publish-acceptance-gate" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("not package push", StringComparison.OrdinalIgnoreCase));

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-public-publish-acceptance-gate" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(audit.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final public publish acceptance gate");
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
