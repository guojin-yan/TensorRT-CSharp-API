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
        Assert.True(import.GetProperty("failedActionRequiredCount").GetInt32() >= 38);
        Assert.True(import.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(import.GetProperty("mappingCount").GetInt32() >= 37);
        Assert.Equal(0, import.GetProperty("existingFileCount").GetInt32());
        Assert.True(import.GetProperty("sha256RequiredFileCount").GetInt32() >= 37);
        Assert.Equal(0, import.GetProperty("sha256ValidFileCount").GetInt32());
        Assert.False(import.GetProperty("rootOutsideRepository").GetBoolean());
        Assert.False(import.GetProperty("requireExistingFiles").GetBoolean());
        Assert.False(import.GetProperty("requireHashMatch").GetBoolean());
        AssertNonProof(import);

        using JsonDocument candidateDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("blocked-owner-real-proof-staging-workspace-candidate", candidate.GetProperty("candidateState").GetString());
        Assert.False(candidate.GetProperty("readyForStrictImport").GetBoolean());
        Assert.False(candidate.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(candidate.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(candidate.GetProperty("mappingCount").GetInt32() >= 37);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-proof-staging-workspace-validation-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("readyForStrictImport").GetBoolean());
        Assert.False(validation.GetProperty("proofCandidateReady").GetBoolean());
        Assert.True(validation.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(validation.GetProperty("mappingCount").GetInt32() >= 37);
        Assert.True(validation.GetProperty("sha256RequiredFileCount").GetInt32() >= 37);
        AssertNonProof(validation);
    }

    [Fact]
    public void StagingWorkspaceImportAllowsDownloadedPublicNupkgEvidenceFiles()
    {
        RunPowerShell("Export-OwnerRealProofStagingWorkspaceContract.ps1");

        string stagingRoot = Path.Combine(
            Path.GetTempPath(),
            "TensorRtSharp-owner-real-proof-staging-" + Guid.NewGuid().ToString("N"));

        try
        {
            using JsonDocument contractDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-contract.json");
            foreach (JsonElement file in contractDocument.RootElement.GetProperty("requiredFiles").EnumerateArray())
            {
                string relativePath = file.GetProperty("relativePath").GetString()!;
                string destination = Path.Combine(
                    stagingRoot,
                    relativePath.Replace('/', Path.DirectorySeparatorChar));
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                File.WriteAllText(destination, $"owner-staging-placeholder:{relativePath}");
            }

            string output = RunPowerShell(
                "Import-OwnerRealProofStagingWorkspace.ps1",
                "-OwnerStagingRoot",
                stagingRoot,
                "-RequireExistingFiles",
                "-RequireHashMatch");
            Assert.Contains("ReadyForStrictImport=True", output, StringComparison.Ordinal);
            RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");

            using JsonDocument importDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-import.json");
            JsonElement import = importDocument.RootElement;

            Assert.Equal("owner-real-proof-staging-workspace-import-ready", import.GetProperty("importState").GetString());
            Assert.True(import.GetProperty("readyForStrictImport").GetBoolean());
            Assert.True(import.GetProperty("rootOutsideRepository").GetBoolean());
            Assert.True(import.GetProperty("requireExistingFiles").GetBoolean());
            Assert.True(import.GetProperty("requireHashMatch").GetBoolean());
            Assert.Equal(import.GetProperty("mappingCount").GetInt32(), import.GetProperty("existingFileCount").GetInt32());
            Assert.Equal(import.GetProperty("sha256RequiredFileCount").GetInt32(), import.GetProperty("sha256ValidFileCount").GetInt32());
            Assert.Equal(0, import.GetProperty("forbiddenPathCount").GetInt32());
            Assert.Equal(0, import.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.False(import.GetProperty("proofCandidateReady").GetBoolean());
            AssertNonProof(import);

            JsonElement[] publicPackageFiles = import.GetProperty("mappingResults")
                .EnumerateArray()
                .Where(static mapping => mapping.GetProperty("sourcePath").GetString()!.EndsWith(".nupkg", StringComparison.OrdinalIgnoreCase))
                .ToArray();
            Assert.Equal(2, publicPackageFiles.Length);
            Assert.All(publicPackageFiles, static mapping => Assert.False(mapping.GetProperty("forbiddenPath").GetBoolean()));
        }
        finally
        {
            RunPowerShell("Import-OwnerRealProofStagingWorkspace.ps1");
            RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");

            if (Directory.Exists(stagingRoot))
            {
                Directory.Delete(stagingRoot, recursive: true);
            }
        }
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
