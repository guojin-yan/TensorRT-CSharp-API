using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionCloseReadinessFromRealInputTests
{
    [Fact]
    public void CloseReadinessFromRealInputRemainsBlockedWithoutRealOwnerProof()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Import-FinalOwnerExecutionRealInput.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputStrictPreflight.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerExecutionCloseReadinessFromRealInput.ps1", "-Strict");

        using JsonDocument readinessDocument = ReadFinalReleaseJson("final-owner-execution-close-readiness-from-real-input.json");
        JsonElement readiness = readinessDocument.RootElement;

        Assert.Equal("final-owner-execution-close-readiness-from-real-input", readiness.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-close-readiness-required", readiness.GetProperty("readinessState").GetString());
        Assert.True(readiness.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(readiness.GetProperty("readinessCheckCount").GetInt32() >= 5);
        Assert.Equal(readiness.GetProperty("readinessCheckCount").GetInt32(), readiness.GetProperty("blockedReadinessCheckCount").GetInt32());
        AssertNonProof(readiness);

        string[] checks = readiness.GetProperty("checks").EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string expected in new[]
        {
            "real-input-strict-preflight-ready",
            "external-clean-consumer-proof-present",
            "post-publish-proof-present",
            "rollback-review-present",
            "final-close-decision-present",
            "release-evidence-classification-clean"
        })
        {
            Assert.Contains(expected, checks);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-close-readiness-from-real-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-close-readiness-from-real-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-close-readiness-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 6);
        Assert.False(validation.GetProperty("externalCleanConsumerProofReady").GetBoolean());
        Assert.False(validation.GetProperty("postPublishProofReady").GetBoolean());
        AssertNonProof(validation);
    }

    [Fact]
    public void ReleaseEvidenceKeepsRealInputChainFailedAndNonProof()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputTemplate.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerExecutionRealInput.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputImport.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerExecutionRealInputStrictPreflight.ps1", "-Strict");
        RunPowerShell("Export-ExternalCleanConsumerExecutionWorkspaceContract.ps1");
        RunPowerShell("Test-ExternalCleanConsumerExecutionWorkspaceContract.ps1", "-Strict");
        RunPowerShell("Export-ExternalCleanConsumerOwnerCommandPack.ps1");
        RunPowerShell("Test-ExternalCleanConsumerOwnerCommandPack.ps1", "-Strict");
        RunPowerShell("Import-ExternalCleanConsumerExecutionResult.ps1");
        RunPowerShell("Test-ExternalCleanConsumerExecutionResult.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerProofResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerExecutionCloseReadinessFromRealInput.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        string bundleText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.json"));

        Assert.Equal(2, bundle.GetProperty("finalOwnerExecutionRealInputStrictPreflightDualPackageRouteProofRouteCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionRealInputStrictPreflightDualPackageRouteProofFieldCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("finalOwnerExecutionRealInputStrictPreflightDualPackageRouteProofReadyFieldCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionRealInputStrictPreflightDualPackageRouteProofPlaceholderFieldCount").GetInt32());

        JsonElement strictPreflightItem = bundle.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "final-owner-execution-real-input-strict-preflight");
        string strictPreflightState = strictPreflightItem.GetProperty("state").GetString()!;
        Assert.Contains("dualPackageRouteProofRoutes=2", strictPreflightState, StringComparison.Ordinal);
        Assert.Contains("dualPackageRouteProofFields=8", strictPreflightState, StringComparison.Ordinal);
        Assert.Contains("dualPackageRouteProofReadyFields=0", strictPreflightState, StringComparison.Ordinal);
        Assert.Contains("dualPackageRouteProofPlaceholders=8", strictPreflightState, StringComparison.Ordinal);

        foreach (string id in new[]
        {
            "final-owner-execution-real-input-template",
            "final-owner-execution-real-input-import",
            "final-owner-execution-real-input-candidate",
            "final-owner-execution-real-input-strict-preflight",
            "final-owner-execution-close-readiness-from-real-input",
            "external-clean-consumer-execution-workspace-contract",
            "external-clean-consumer-owner-command-pack",
            "external-clean-consumer-execution-result-import",
            "external-clean-consumer-execution-result-candidate",
            "post-publish-clean-consumer-proof-result-import",
            "post-publish-clean-consumer-proof-result-candidate"
        })
        {
            JsonElement item = bundle.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.Contains("not runtime proof", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not post-publish proof", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not publish approval", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not release close approval", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains(id, bundleText, StringComparison.Ordinal);
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
