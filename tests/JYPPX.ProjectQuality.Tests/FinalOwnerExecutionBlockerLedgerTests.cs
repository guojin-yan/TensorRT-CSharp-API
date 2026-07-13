using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionBlockerLedgerTests
{
    [Fact]
    public void BlockerLedgerGroupsRemainingOwnerInputBlockersWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Test-FinalOwnerExecutionInputSkeleton.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerExecutionBlockerLedger.ps1");
        RunPowerShell("Test-FinalOwnerExecutionBlockerLedger.ps1", "-Strict");

        using JsonDocument ledgerDocument = ReadFinalReleaseJson("final-owner-execution-blocker-ledger.json");
        JsonElement ledger = ledgerDocument.RootElement;

        Assert.Equal("final-owner-execution-blocker-ledger", ledger.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", ledger.GetProperty("ledgerState").GetString());
        Assert.True(ledger.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(ledger.GetProperty("performsPublish").GetBoolean());
        Assert.False(ledger.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(ledger.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(ledger.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ledger.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(ledger.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(ledger.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(ledger.GetProperty("isReleaseCloseProof").GetBoolean());

        Assert.True(ledger.GetProperty("blockerCount").GetInt32() >= 90);
        Assert.True(ledger.GetProperty("remainingBlockerCount").GetInt32() >= 90);
        Assert.Equal(0, ledger.GetProperty("readyForImportCandidateCount").GetInt32());

        string[] categories = ledger.GetProperty("categories").EnumerateArray().Select(static item => item.GetProperty("category").GetString()!).ToArray();
        foreach (string category in new[]
        {
            "missing owner input",
            "placeholder",
            "path missing",
            "SHA256 invalid",
            "strict validator not run",
            "strict validator failed",
            "ready for import candidate"
        })
        {
            Assert.Contains(category, categories);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-blocker-ledger-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-blocker-ledger-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
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
