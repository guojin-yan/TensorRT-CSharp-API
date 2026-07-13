using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class BridgePackageRuntimeConsumerShortPathTests
{
    [Fact]
    public void RuntimeConsumerScriptDefaultsToShortRestorePathForLongRuntimePackageIds()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-BridgePackageRuntimeConsumer.ps1"));

        Assert.Contains("function Get-ShortRuntimeConsumerRoot", script, StringComparison.Ordinal);
        Assert.Contains("\"jybr\"", script, StringComparison.Ordinal);
        Assert.Contains("$consumerRoot = Get-ShortRuntimeConsumerRoot -RuntimeKey $SourceRuntimeKey -Rid $rid", script, StringComparison.Ordinal);
        Assert.Contains("<RestorePackagesPath>$restorePackagesPath</RestorePackagesPath>", script, StringComparison.Ordinal);
        Assert.Contains("Runtime proof consumer root must be outside the source repository", script, StringComparison.Ordinal);
        Assert.Contains("-replace \"win-x64-\"", script, StringComparison.Ordinal);
        Assert.Contains("if ($runtimeToken.Length -gt 28)", script, StringComparison.Ordinal);
        Assert.DoesNotContain("jyppx-bridge-package-runtime-consumer\"", script, StringComparison.Ordinal);
    }
}
