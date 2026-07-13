using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionRealInputImportTests
{
    [Fact]
    public void RealInputImportCreatesBlockedCandidateWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputTemplate.ps1", "-Strict");
        RunPowerShell("Import-FinalOwnerExecutionRealInput.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputImport.ps1", "-Strict");

        using JsonDocument importDocument = ReadFinalReleaseJson("final-owner-execution-real-input-import.json");
        JsonElement import = importDocument.RootElement;
        Assert.Equal("final-owner-execution-real-input-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", import.GetProperty("importState").GetString());
        Assert.Equal(41, import.GetProperty("overlayFieldCount").GetInt32());
        Assert.Equal(0, import.GetProperty("readyFieldCount").GetInt32());
        Assert.True(import.GetProperty("placeholderFieldCount").GetInt32() >= 39);
        Assert.True(import.GetProperty("invalidSha256FieldCount").GetInt32() >= 8);
        Assert.False(import.GetProperty("readyForImport").GetBoolean());
        AssertNonProof(import);

        using JsonDocument candidateDocument = ReadFinalReleaseJson("final-owner-execution-real-input-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("final-owner-execution-real-input-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", candidate.GetProperty("candidateState").GetString());
        Assert.True(candidate.GetProperty("fieldResultCount").GetInt32() >= 39);
        Assert.Equal(0, candidate.GetProperty("readyFieldCount").GetInt32());
        Assert.False(candidate.GetProperty("readyForImport").GetBoolean());
        AssertNonProof(candidate);

        JsonElement[] fieldResults = candidate.GetProperty("fieldResults").EnumerateArray().ToArray();
        Assert.Contains(fieldResults, item => item.GetProperty("fieldPath").GetString() == "cleanConsumer.projectRoot" && !item.GetProperty("placeholderReplaced").GetBoolean());
        Assert.Contains(fieldResults, item => item.GetProperty("kind").GetString() == "sha256" && !item.GetProperty("sha256FormatValid").GetBoolean());
        Assert.Contains(fieldResults, item => item.GetProperty("kind").GetString() == "path" && !item.GetProperty("pathExists").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-real-input-import-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-real-input-import-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("fieldResultCount").GetInt32() >= 39);
        Assert.Equal(0, validation.GetProperty("readyFieldCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
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
