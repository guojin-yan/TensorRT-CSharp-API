using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ExternalCleanConsumerExecutionWorkspaceContractTests
{
    [Fact]
    public void WorkspaceContractRequiresRepositoryExternalEvidenceAndStaysNonProof()
    {
        RunPowerShell("Export-ExternalCleanConsumerExecutionWorkspaceContract.ps1");
        RunPowerShell("Test-ExternalCleanConsumerExecutionWorkspaceContract.ps1", "-Strict");

        using JsonDocument contractDocument = ReadFinalReleaseJson("external-clean-consumer-execution-workspace-contract.json");
        JsonElement contract = contractDocument.RootElement;

        Assert.Equal("external-clean-consumer-execution-workspace-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-external-clean-consumer-workspace-contract-required", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(contract.GetProperty("passed").GetBoolean());
        AssertNonProof(contract);

        string text = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-clean-consumer-execution-workspace-contract.json"));
        foreach (string expected in new[]
        {
            "repository-external",
            "outside this repository",
            "ProjectReference",
            "local feed",
            "direct .nupkg",
            "packageSourceUrl",
            "restoreLogSha256",
            "buildLogSha256",
            "runLogSha256",
            "smokeStdoutSha256",
            "smokeStderrSha256",
            "nativeAssetListingSha256",
            "managedPackageSha256",
            "runtimePackageSha256",
            "hostMetadata"
        })
        {
            Assert.Contains(expected, text, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("external-clean-consumer-execution-workspace-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("external-clean-consumer-execution-workspace-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("external-clean-consumer-execution-workspace-contract-ready-non-proof", validation.GetProperty("validationState").GetString());
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

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
