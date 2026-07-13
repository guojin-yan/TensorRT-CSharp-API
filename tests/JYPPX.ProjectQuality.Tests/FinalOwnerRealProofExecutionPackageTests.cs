using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerRealProofExecutionPackageTests
{
    [Fact]
    public void FinalOwnerRealProofExecutionPackageProvidesOwnerSequenceWithoutBecomingProof()
    {
        RunPowerShell("Export-FinalOwnerRealProofExecutionPackage.ps1");
        RunPowerShell("Test-FinalOwnerRealProofExecutionPackage.ps1", "-Strict");

        using JsonDocument packageDocument = ReadFinalReleaseJson("final-owner-real-proof-execution-package.json");
        JsonElement package = packageDocument.RootElement;

        Assert.Equal("final-owner-real-proof-execution-package", package.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-proof-execution-required", package.GetProperty("packageState").GetString());
        Assert.True(package.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(package.GetProperty("passed").GetBoolean());
        Assert.True(package.GetProperty("executionStepCount").GetInt32() >= 14);
        AssertNonProof(package);

        string json = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-real-proof-execution-package.json"));
        foreach (string expected in new[]
        {
            "create-repository-external-clean-consumer-workspace",
            "restore-from-real-public-package-source",
            "run-external-clean-consumer-import-and-strict-validator",
            "download-public-published-packages",
            "run-post-publish-import-and-strict-validator",
            "fill-rollback-review",
            "fill-final-close-decision",
            "failedBlockerCount=0",
            "pre-publish smoke reused as post-publish proof"
        })
        {
            Assert.Contains(expected, json, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-real-proof-execution-package-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-real-proof-execution-package-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("final-owner-real-proof-execution-package-ready-non-proof", validation.GetProperty("validationState").GetString());
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
