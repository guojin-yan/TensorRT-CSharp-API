using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class WindowsBridgePackageMatrixTests
{
    [Fact]
    public void MatrixRunnerBuildsAndValidatesBridgeOnlyCandidatesWithoutPublishing()
    {
        string script = ReadSource("eng", "Invoke-WindowsBridgePackageMatrix.ps1");

        foreach (string marker in new[]
        {
            "role -eq \"bridge\"",
            "platform -eq \"windows\"",
            "Invoke-LocalSplitRuntimePackage.ps1",
            "-SkipConsumerValidation",
            "Test-ExternalVendorRuntimePackagePolicy.ps1",
            "-RequireExactPackageSet",
            "Test-BridgePackageConsumer.ps1",
            "bridge-package-consumer-validation-summary.json",
            "windows-bridge-package-matrix.json",
            "local-windows-bridge-package-matrix",
            "vendorRuntimeBundled = $false",
            "isRuntimeExecutionProof = $false",
            "isPackageConsumerRuntimeProof = $false",
            "canPublishPublicly = $false",
            "publicationExecuted = $false",
        })
        {
            Assert.Contains(marker, script, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("git tag", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ManifestDefinesExactlySixWindowsBridgePackagesWithOneOwnedAssetEach()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "pack",
            "runtime-split",
            "split-runtime-packages.manifest.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));

        JsonElement[] packages = document.RootElement.GetProperty("packages")
            .EnumerateArray()
            .Where(static package =>
                package.GetProperty("role").GetString() == "bridge" &&
                package.GetProperty("platform").GetString() == "windows")
            .ToArray();

        Assert.Equal(6, packages.Length);
        Assert.Equal(6, packages.Select(static package => package.GetProperty("sourceRuntimeKey").GetString()).Distinct().Count());
        foreach (JsonElement package in packages)
        {
            Assert.EndsWith(".Bridge", package.GetProperty("packageId").GetString(), StringComparison.Ordinal);
            Assert.Equal("win-x64", package.GetProperty("rid").GetString());
            Assert.Equal(
                new[] { "jyppxtrtbridge.dll" },
                package.GetProperty("assets").EnumerateArray().Select(static asset => asset.GetString()).ToArray());
        }
    }

    [Fact]
    public void EngineeringDocsClassifyMatrixRunnerAsSupportedLocalOnlyEntrypoint()
    {
        string engineeringReadme = ReadSource("eng", "README.md");
        string bridgeReadme = ReadSource("pack", "runtime-split", "README.md");

        Assert.Contains("Invoke-WindowsBridgePackageMatrix.ps1", engineeringReadme, StringComparison.Ordinal);
        Assert.Contains("六组 bridge-only 本地候选", engineeringReadme, StringComparison.Ordinal);
        Assert.Contains("不执行上传、tag 或 Release", engineeringReadme, StringComparison.Ordinal);
        Assert.Contains("complete Windows bridge matrix without publishing", bridgeReadme, StringComparison.Ordinal);
        Assert.Contains("local candidate evidence only", bridgeReadme, StringComparison.Ordinal);
        Assert.Contains("PublicationExecuted=False", bridgeReadme, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
