using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerLinuxProofPackTests
{
    [Fact]
    public void PackageConsumerRuntimeProofPackExportsBlockedOwnerExecutionPlan()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofExecutionPack.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofPack.ps1"));

        Assert.Contains("Package consumer runtime proof execution pack written", output, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-action-required", validationOutput, StringComparison.Ordinal);

        JsonElement pack = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-execution-pack.json"));
        Assert.Equal("package-consumer-runtime-proof-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("packState").GetString());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(pack.GetProperty("noProjectReferenceRequired").GetBoolean());
        Assert.True(pack.GetProperty("missingOwnerInputCount").GetInt32() >= 12);
        Assert.Contains("outside-repository", pack.GetProperty("cleanConsumerRoot").GetString()!, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("dependency probes", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", pack.GetProperty("upstreamValidatorCommand").GetString()!, StringComparison.Ordinal);
        Assert.Contains(pack.GetProperty("requiredCommands").EnumerateArray(), static item => item.GetString()!.Contains("dotnet restore", StringComparison.Ordinal));
        Assert.Contains(pack.GetProperty("missingOwnerInputs").EnumerateArray(), static item => item.GetString()!.Contains("no ProjectReference", StringComparison.Ordinal));

        JsonElement validation = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-pack-validation.json"));
        Assert.Equal("package-consumer-runtime-proof-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.False(validation.GetProperty("cleanConsumerRootInsideRepository").GetBoolean());
        Assert.False(validation.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("ProjectReference", validation.GetProperty("boundary").GetString()!, StringComparison.Ordinal);
    }

    [Fact]
    public void LinuxRunnerProofPackExportsBlockedOwnerExecutionPlan()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerProofExecutionPack.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-LinuxRunnerProofPack.ps1"));

        Assert.Contains("Linux runner proof execution pack written", output, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-action-required", validationOutput, StringComparison.Ordinal);

        JsonElement pack = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "linux-runner-proof-execution-pack.json"));
        Assert.Equal("linux-runner-proof-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("packState").GetString());
        Assert.Equal("linux-x64", pack.GetProperty("targetArch").GetString());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteLinuxRunnerProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(pack.GetProperty("missingOwnerInputCount").GetInt32() >= 12);
        Assert.Contains("WSL", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("Test-LinuxRunnerEvidenceRecord.ps1", pack.GetProperty("upstreamValidatorCommand").GetString()!, StringComparison.Ordinal);
        Assert.Contains(pack.GetProperty("requiredCommands").EnumerateArray(), static item => item.GetString()!.Contains("nvidia-smi", StringComparison.Ordinal));
        Assert.Contains(pack.GetProperty("missingOwnerInputs").EnumerateArray(), static item => item.GetString()!.Contains("GPU name", StringComparison.Ordinal));

        JsonElement validation = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "linux-runner-proof-pack-validation.json"));
        Assert.Equal("linux-runner-proof-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.False(validation.GetProperty("canPromoteLinuxRunnerProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Windows dry-run", validation.GetProperty("boundary").GetString()!, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseMatricesAggregatePackageAndLinuxProofPacksWithoutUnlockingRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-LinuxRunnerProofPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseFreezeFinalVerification.ps1"));

        JsonElement matrix = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-runtime-proof-execution-matrix.json"));
        Assert.Equal("blocked-owner-action-required", matrix.GetProperty("packageConsumerProofPackState").GetString());
        Assert.Equal("blocked-owner-action-required", matrix.GetProperty("linuxRunnerProofPackState").GetString());
        Assert.False(matrix.GetProperty("packageConsumerProofPackCanPromote").GetBoolean());
        Assert.False(matrix.GetProperty("linuxRunnerProofPackCanPromote").GetBoolean());
        Assert.False(matrix.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(matrix.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-execution-pack.json", matrix.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("artifacts/final-release/linux-runner-proof-execution-pack.json", matrix.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains(matrix.GetProperty("proofItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "package-consumer-proof-pack");
        Assert.Contains(matrix.GetProperty("proofItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "linux-runner-proof-pack");

        JsonElement freeze = ReadJsonRoot(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.json"));
        Assert.Equal("blocked-real-proof-required", freeze.GetProperty("verificationState").GetString());
        Assert.Equal(5, freeze.GetProperty("releaseBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-action-required", freeze.GetProperty("packageConsumerProofPackState").GetString());
        Assert.Equal("blocked-owner-action-required", freeze.GetProperty("linuxRunnerProofPackState").GetString());
        Assert.False(freeze.GetProperty("packageConsumerProofPackCanPromote").GetBoolean());
        Assert.False(freeze.GetProperty("linuxRunnerProofPackCanPromote").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(freeze.GetProperty("releaseProofFinalAuditItems").EnumerateArray(), static item => item.GetProperty("proofId").GetString() == "package-consumer-proof-pack");
        Assert.Contains(freeze.GetProperty("releaseProofFinalAuditItems").EnumerateArray(), static item => item.GetProperty("proofId").GetString() == "linux-runner-proof-pack");
    }

    private static JsonElement ReadJsonRoot(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}
