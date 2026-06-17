using System.Text.Json;
using System.Diagnostics;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseAutomationTests
{
    private static readonly string[] WindowsRuntimeKeys =
    [
        "win-x64-trt8.6-cuda11.8-cudnn8.9",
        "win-x64-trt8.6-cuda12.1-cudnn8.9",
        "win-x64-trt10.11-cuda11.8-cudnn8.9",
        "win-x64-trt10.11-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda12.9-cudnn9.22",
        "win-x64-trt11.0-cuda13.2-cudnn9.22",
    ];

    [Fact]
    public void RuntimeWorkflowsDefaultToCompleteHostedMatrices()
    {
        string releaseBundle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-bundle.yml"));
        string runtimeWindows = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-windows.yml"));
        string runtimeLinux = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml"));

        foreach (string key in WindowsRuntimeKeys)
        {
            Assert.Contains(key, releaseBundle, StringComparison.Ordinal);
            Assert.Contains(key, runtimeWindows, StringComparison.Ordinal);
        }

        Assert.Contains("default: hosted-all", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("default: auto", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("runtime_key_set", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("runner_mode", runtimeLinux, StringComparison.Ordinal);
        Assert.Contains("fromJson(matrix.runsOnJson)", runtimeLinux, StringComparison.Ordinal);
    }

    [Fact]
    public void RootReadmeRemoteRuntimeExamplesDoNotNarrowWindowsToOneCombination()
    {
        string english = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string chinese = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string samples = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));

        Assert.DoesNotContain("-WindowsRuntimeKeys win-x64-trt11.0-cuda12.9-cudnn9.22", english, StringComparison.Ordinal);
        Assert.DoesNotContain("-f windows_runtime_keys=win-x64-trt11.0-cuda12.9-cudnn9.22", chinese, StringComparison.Ordinal);
        Assert.DoesNotContain("-WindowsRuntimeKeys win-x64-trt11.0-cuda12.9-cudnn9.22", samples, StringComparison.Ordinal);
        Assert.Contains("omit `-WindowsRuntimeKeys` to use the full six-combination Windows matrix", english, StringComparison.Ordinal);
        Assert.Contains("不传 `windows_runtime_keys`，使用默认 Windows 6 组合矩阵", chinese, StringComparison.Ordinal);
        Assert.Contains("-WindowsRuntimeKeys <runtime-key>", samples, StringComparison.Ordinal);
    }

    [Fact]
    public void LinuxTargetCatalogKeepsUbuntuAndFutureArchitectureLinesExplicit()
    {
        string path = Path.Combine(RepositoryPaths.Root, "pack", "runtime", "linux-runtime-targets.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));

        JsonElement[] targets = document.RootElement.GetProperty("targets").EnumerateArray().ToArray();
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu22.04-x64-hosted");
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu24.04-x64-hosted");
        Assert.Contains(targets, static target => target.GetProperty("target").GetString() == "ubuntu20.04-x64-self-hosted");

        JsonElement[] futureTargets = document.RootElement.GetProperty("futureTargets").EnumerateArray().ToArray();
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-arm64-sbsa");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "linux-jetson-l4t");
        Assert.Contains(futureTargets, static target => target.GetProperty("target").GetString() == "non-ubuntu-linux");
    }

    [Fact]
    public void ReleasePublicationAuditCanSkipPrerequisiteNoiseSeparatelyFromInventory()
    {
        string workflow = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "release-publication-audit.yml"));

        Assert.Contains("check_remote_release_prerequisites", workflow, StringComparison.Ordinal);
        Assert.Contains("check_runner_availability", workflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ inputs.check_remote_release_prerequisites }}", workflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ inputs.check_runner_availability }}", workflow, StringComparison.Ordinal);
        Assert.Contains("include_release_readiness", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void RemoteReleaseBundleDryRunCarriesStableDependencyVersionMaps()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Invoke-RemoteReleaseBundle.ps1");
        string output = RunPowerShell(
            script,
            "-Version", "4.0.7000",
            "-RuntimeVersion", "4.0.7000",
            "-RunWindowsRuntimePackaging",
            "-WindowsSplitPackageRoles", "bridge,collection",
            "-RunLinuxRuntimePackaging",
            "-LinuxRuntimeKeySet", "hosted-all",
            "-LinuxSplitPackageRoles", "bridge,collection",
            "-WindowsCudaCudnnPackageVersionMap", "win-x64-*=4.0.6156",
            "-WindowsTensorRtPackageVersionMap", "win-x64-*=4.0.6156",
            "-LinuxCudaCudnnPackageVersionMap", "linux-x64-ubuntu22.04-*=4.0.6167;linux-x64-ubuntu24.04-*=4.0.6169",
            "-LinuxTensorRtPackageVersionMap", "linux-x64-ubuntu22.04-*=4.0.6167;linux-x64-ubuntu24.04-*=4.0.6169",
            "-LinuxIncludeMetaPackage",
            "-WindowsIncludeMetaPackage",
            "-PublishRuntimeToGitHubPackages", "true",
            "-DryRun");

        Assert.Contains("gh workflow run release-bundle.yml", output, StringComparison.Ordinal);
        Assert.Contains("release_config_json=", output, StringComparison.Ordinal);
        Assert.Contains("windows_cuda_cudnn_package_version_map", output, StringComparison.Ordinal);
        Assert.Contains("windows_tensorrt_package_version_map", output, StringComparison.Ordinal);
        Assert.Contains("linux_cuda_cudnn_package_version_map", output, StringComparison.Ordinal);
        Assert.Contains("linux_tensorrt_package_version_map", output, StringComparison.Ordinal);
        Assert.Contains("hosted-all", output, StringComparison.Ordinal);
        Assert.Contains("bridge,collection", output, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("hosted-all", "hosted", 9, "linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9", "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22")]
    [InlineData("ubuntu24-hosted", "hosted", 3, "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22", "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22")]
    [InlineData("self-hosted-ubuntu20", "self-hosted", 3, "linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9", "linux-x64-ubuntu20.04-trt10.11-cuda11.8-cudnn8.9")]
    public void LinuxRuntimeKeySetsResolveExpectedPackageLines(
        string keySet,
        string runnerMode,
        int expectedCount,
        string expectedFirstKey,
        string expectedSecondKey)
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeKeySet.ps1");
        string output = RunPowerShell(
            script,
            "-Platform", "linux",
            "-RuntimeKeySet", keySet,
            "-RunnerMode", runnerMode,
            "-OutputFormat", "json");

        string[] keys = JsonSerializer.Deserialize<string[]>(output)!;

        Assert.Equal(expectedCount, keys.Length);
        Assert.Contains(expectedFirstKey, keys);
        Assert.Contains(expectedSecondKey, keys);
    }

    [Fact]
    public void Ubuntu20SelfHostedMatrixUsesOfficialAptDependencyPreparation()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeMatrix.ps1");
        string output = RunPowerShell(
            script,
            "-Platform", "linux",
            "-RuntimeKey", "linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9",
            "-RunnerMode", "self-hosted");

        using JsonDocument document = JsonDocument.Parse(output);
        JsonElement entry = Assert.Single(document.RootElement.EnumerateArray());

        Assert.Equal("self-hosted", entry.GetProperty("runnerMode").GetString());
        Assert.Equal("apt", entry.GetProperty("nvidiaDependencyMode").GetString());
        Assert.Contains("ubuntu-20.04", entry.GetProperty("runsOnJson").GetString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("arm64-sbsa", "linux-arm64-sbsa")]
    [InlineData("jetson-l4t", "linux-jetson-l4t")]
    [InlineData("non-ubuntu", "non-ubuntu-linux")]
    public void FutureLinuxKeySetsFailInsteadOfDispatchingUnsupportedPackages(string keySet, string targetName)
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Resolve-RuntimeKeySet.ps1");
        (int exitCode, string output) = RunPowerShellAllowFailure(
            script,
            "-Platform", "linux",
            "-RuntimeKeySet", keySet,
            "-RunnerMode", "self-hosted",
            "-OutputFormat", "json");

        Assert.NotEqual(0, exitCode);
        Assert.Contains("future package line", output, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(targetName, output, StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        (int exitCode, string output) = RunPowerShellAllowFailure(scriptPath, arguments);

        Assert.True(exitCode == 0, $"PowerShell command failed with exit code {exitCode}:{Environment.NewLine}{output}");
        return output.Trim();
    }

    private static (int ExitCode, string Output) RunPowerShellAllowFailure(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
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

        return (process.ExitCode, output + error);
    }
}
