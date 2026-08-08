using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ExternalCleanConsumerOwnerCommandPackTests
{
    [Fact]
    public void OwnerCommandPackProvidesExecutableGuidanceWithoutBecomingProof()
    {
        RunPowerShell("Export-ExternalCleanConsumerOwnerCommandPack.ps1");
        RunPowerShell("Test-ExternalCleanConsumerOwnerCommandPack.ps1", "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("external-clean-consumer-owner-command-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("external-clean-consumer-owner-command-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-external-clean-consumer-owner-command-pack-required", pack.GetProperty("packState").GetString());
        Assert.True(pack.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(pack.GetProperty("passed").GetBoolean());
        Assert.True(pack.GetProperty("stepCount").GetInt32() >= 10);
        AssertNonProof(pack);

        string text = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-clean-consumer-owner-command-pack.json"));
        foreach (string expected in new[]
        {
            "dotnet new",
            "dotnet add",
            "dotnet restore",
            "dotnet build",
            "dotnet run",
            "Get-FileHash",
            "native-assets",
            "host-metadata",
            "ProjectReference",
            "local feed",
            "direct .nupkg",
            "pre-publish smoke reused as post-publish proof"
        })
        {
            Assert.Contains(expected, text, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("external-clean-consumer-owner-command-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("external-clean-consumer-owner-command-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("external-clean-consumer-owner-command-pack-ready-non-proof", validation.GetProperty("validationState").GetString());
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
