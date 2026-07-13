using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofStagingWorkspaceImportTests
{
    [Fact]
    public void StagingWorkspaceImportDefaultsToBlockedAndRequiresOwnerFiles()
    {
        RunPowerShell("Import-OwnerRealProofStagingWorkspace.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");

        using JsonDocument importDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-import.json");
        JsonElement import = importDocument.RootElement;

        Assert.Equal("owner-real-proof-staging-workspace-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-staging-workspace-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("readyForStrictImport").GetBoolean());
        Assert.False(import.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(import.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(import.GetProperty("failedActionRequiredCount").GetInt32() >= 10);
        AssertNonProof(import);

        using JsonDocument candidateDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("blocked-owner-real-proof-staging-workspace-candidate", candidate.GetProperty("candidateState").GetString());
        Assert.False(candidate.GetProperty("readyForStrictImport").GetBoolean());
        Assert.False(candidate.GetProperty("proofCandidateReady").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-proof-staging-workspace-validation-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("readyForStrictImport").GetBoolean());
        Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
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
