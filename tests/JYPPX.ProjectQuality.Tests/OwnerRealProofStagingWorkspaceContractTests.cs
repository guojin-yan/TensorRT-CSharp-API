using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofStagingWorkspaceContractTests
{
    [Fact]
    public void StagingWorkspaceContractDefinesOwnerLayoutWithoutReadingProofFiles()
    {
        RunPowerShell("Export-OwnerRealProofStagingWorkspaceContract.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspaceContract.ps1", "-Strict");

        using JsonDocument contractDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-contract.json");
        JsonElement contract = contractDocument.RootElement;

        Assert.Equal("owner-real-proof-staging-workspace-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-staging-workspace-contract-required", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(contract.GetProperty("passed").GetBoolean());
        Assert.True(contract.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(contract.GetProperty("requiredFileCount").GetInt32() >= 37);
        Assert.True(contract.GetProperty("sha256RequiredFileCount").GetInt32() >= 37);
        Assert.True(contract.GetProperty("forbiddenSubstituteCount").GetInt32() >= 300);
        Assert.True(contract.GetProperty("externalWorkspaceRequired").GetBoolean());
        AssertNonProof(contract);

        string[] paths = contract.GetProperty("requiredFiles")
            .EnumerateArray()
            .Select(static file => file.GetProperty("relativePath").GetString()!)
            .ToArray();

        foreach (string expected in new[]
        {
            "public-package/nuget-managed-package.nupkg",
            "public-package/github-runtime-package.nupkg",
            "public-package/downloaded-packages.json",
            "public-package/download-transcript.log",
            "external-clean-consumer/consumer.csproj",
            "external-clean-consumer/package-source.json",
            "external-clean-consumer/resolved-packages.json",
            "external-clean-consumer/restore.log",
            "external-clean-consumer/build.log",
            "external-clean-consumer/smoke.stdout.log",
            "external-clean-consumer/smoke.stderr.log",
            "external-clean-consumer/merged-transcript.log",
            "external-clean-consumer/native-assets.json",
            "external-clean-consumer/host-metadata.json",
            "external-clean-consumer/no-local-substitute-confirmation.json",
            "yolovision/model.onnx",
            "yolovision/asset-manifest.json",
            "yolovision/output.json",
            "yolovision/runtime-transcript.log",
            "yolovision/real-model-execution-confirmation.json",
            "article-publication/article-proof-records.json",
            "article-publication/article-proof-manifest.json",
            "article-publication/screenshots.zip",
            "release-close/release-evidence-bundle.sha256",
            "release-close/classification-audit.sha256",
            "release-close/post-publish-proof.sha256",
            "release-close/rollback-review.json",
            "release-close/final-close-decision.json",
            "release-close/known-limitations.json"
        })
        {
            Assert.Contains(expected, paths);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-proof-staging-workspace-contract-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(validation.GetProperty("requiredFileCount").GetInt32() >= 37);
        Assert.True(validation.GetProperty("sha256RequiredFileCount").GetInt32() >= 37);
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
