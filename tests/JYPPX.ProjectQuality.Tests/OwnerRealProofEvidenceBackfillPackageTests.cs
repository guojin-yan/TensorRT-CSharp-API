using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofEvidenceBackfillPackageTests
{
    [Fact]
    public void BackfillPackageMapsOwnerEvidenceGapsWithoutBecomingProof()
    {
        RunPowerShell("Export-FinalOwnerRealProofGapMatrix.ps1");
        RunPowerShell("Test-FinalOwnerRealProofGapMatrix.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealProofEvidenceBackfillPackage.ps1");
        RunPowerShell("Test-OwnerRealProofEvidenceBackfillPackage.ps1", "-Strict");

        using JsonDocument packageDocument = ReadFinalReleaseJson("owner-real-proof-evidence-backfill-package.json");
        JsonElement package = packageDocument.RootElement;

        Assert.Equal("owner-real-proof-evidence-backfill-package", package.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-evidence-backfill-required", package.GetProperty("packageState").GetString());
        Assert.True(package.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(package.GetProperty("passed").GetBoolean());
        Assert.True(package.GetProperty("backfillItemCount").GetInt32() >= 10);
        AssertNonProof(package);

        string json = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-real-proof-evidence-backfill-package.json"));
        Assert.Contains("external-clean-consumer-execution-result.owner.json", json, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-proof-result.owner.json", json, StringComparison.Ordinal);
        Assert.Contains("final-owner-rollback-review.owner.json", json, StringComparison.Ordinal);
        Assert.Contains("final-owner-close-decision.owner.json", json, StringComparison.Ordinal);
        Assert.Contains("Import-ExternalCleanConsumerExecutionResult.ps1", json, StringComparison.Ordinal);
        Assert.Contains("Import-PostPublishCleanConsumerProofResult.ps1", json, StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-evidence-backfill-package-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-proof-evidence-backfill-package-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("passed").GetBoolean());
        AssertNonProof(validation);
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
