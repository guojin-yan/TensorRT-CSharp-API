using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionInputPreflightTests
{
    [Fact]
    public void InputPreflightRejectsPlaceholdersAndForbiddenSubstitutesWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Test-FinalOwnerExecutionInputPreflight.ps1", "-Strict");

        using JsonDocument preflightDocument = ReadFinalReleaseJson("final-owner-execution-input-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;

        Assert.Equal("final-owner-execution-input-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", preflight.GetProperty("preflightState").GetString());
        Assert.True(preflight.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(preflight.GetProperty("readyForImport").GetBoolean());
        Assert.False(preflight.GetProperty("performsPublish").GetBoolean());
        Assert.False(preflight.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(preflight.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(preflight.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(preflight.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(preflight.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(preflight.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(preflight.GetProperty("isReleaseCloseProof").GetBoolean());

        Assert.True(preflight.GetProperty("checkedFieldCount").GetInt32() >= 39);
        Assert.True(preflight.GetProperty("findingCount").GetInt32() >= 50);
        Assert.Equal(0, preflight.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(preflight.GetProperty("failedActionRequiredCount").GetInt32() >= 50);

        string text = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-execution-input-preflight.json"));
        Assert.Contains("placeholder", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("SHA256 invalid", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("strict validator not run", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("local feed", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ProjectReference", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("direct .nupkg", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("pre-publish smoke reused as post-publish proof", text, StringComparison.OrdinalIgnoreCase);
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
