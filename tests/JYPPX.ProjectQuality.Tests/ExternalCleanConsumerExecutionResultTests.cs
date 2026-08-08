using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ExternalCleanConsumerExecutionResultTests
{
    [Fact]
    public void DefaultExecutionResultImportRemainsBlockedAndFailOnNotProofRejectsIt()
    {
        RunPowerShell("Import-ExternalCleanConsumerExecutionResult.ps1");
        RunPowerShell("Test-ExternalCleanConsumerExecutionResult.ps1", "-Strict");

        using JsonDocument importDocument = ReadFinalReleaseJson("external-clean-consumer-execution-result-import.json");
        JsonElement import = importDocument.RootElement;

        Assert.Equal("external-clean-consumer-execution-result-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-external-clean-consumer-execution-result-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(import.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(import.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(import.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(import.GetProperty("isRuntimeExecutionProof").GetBoolean());

        using JsonDocument candidateDocument = ReadFinalReleaseJson("external-clean-consumer-execution-result-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("blocked-external-clean-consumer-runtime-proof-candidate", candidate.GetProperty("candidateState").GetString());
        Assert.False(candidate.GetProperty("proofCandidateReady").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

        string failOutput = RunPowerShellExpectFailure("Test-ExternalCleanConsumerExecutionResult.ps1", "-FailOnNotProof");
        Assert.Contains("not proof-ready", failOutput, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("external-clean-consumer-execution-result-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("external-clean-consumer-execution-result-validation-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        ProcessResult result = RunPowerShellRaw(scriptName, arguments);
        Assert.True(result.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{result.Stdout}{Environment.NewLine}{result.Stderr}");
        return result.Stdout;
    }

    private static string RunPowerShellExpectFailure(string scriptName, params string[] arguments)
    {
        ProcessResult result = RunPowerShellRaw(scriptName, arguments);
        Assert.NotEqual(0, result.ExitCode);
        return result.Stdout + result.Stderr;
    }

    private static ProcessResult RunPowerShellRaw(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = PowerShellHost.ResolveExecutable(),
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
        return new ProcessResult(process.ExitCode, stdout, stderr);
    }

    private sealed record ProcessResult(int ExitCode, string Stdout, string Stderr);
}
