using System.Text.Json;
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

        foreach (string key in WindowsRuntimeKeys)
        {
            Assert.Contains(key, releaseBundle, StringComparison.Ordinal);
            Assert.Contains(key, runtimeWindows, StringComparison.Ordinal);
        }

        Assert.Contains("default: hosted-all", releaseBundle, StringComparison.Ordinal);
        Assert.Contains("default: auto", File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".github", "workflows", "runtime-linux.yml")), StringComparison.Ordinal);
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
}
