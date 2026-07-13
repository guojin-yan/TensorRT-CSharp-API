using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class VendorRuntimeAssetMaterializationTests
{
    [Fact]
    public void MaterializeScriptCopiesOnlyManifestRuntimeDllsFromArchivesOrDirectories()
    {
        string script = ReadSource("eng", "Materialize-WindowsVendorRuntimeAssets.ps1");

        Assert.Contains("RuntimePackageKey", script);
        Assert.Contains("pack\\runtime\\runtime-packages.manifest.json", script);
        Assert.Contains("Resolve-RuntimeRoots.ps1", script);
        Assert.Contains("tensorRtFiles", script);
        Assert.Contains("cudnnFiles", script);
        Assert.Contains("Test-EntryAllowedForKind", script);
        Assert.Contains("TensorRT-$($package.tensorRtVersion)/bin/", script);
        Assert.Contains("cudnn_cuda$($package.cudaVersion)/libcudnn/bin/$($package.cudaVersion)/x64/", script);
        Assert.Contains("Get-SourceMatches", script);
        Assert.Contains("fileName -like $fileNamePattern", script);
        Assert.Contains("if ($entryPath -notmatch '\\.dll$')", script);
        Assert.Contains("tar -tf", script);
        Assert.Contains("tar -xf", script);
        Assert.Contains("DryRun", script);
        Assert.Contains("dry-run-ready", script);
        Assert.Contains("skipped-existing", script);
        Assert.Contains("missingExpectedCount", script);
        Assert.Contains("vendor-runtime-assets-summary.json", script);
        Assert.Contains("vendor-runtime-assets-summary.md", script);
        Assert.DoesNotContain(".lib$", script);
        Assert.DoesNotContain("Start-Process", script);
    }

    [Fact]
    public void RuntimeReadinessPointsVendorBlockersAtMaterializationPreflight()
    {
        string script = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");

        Assert.Contains("Materialize-WindowsVendorRuntimeAssets.ps1", script);
        Assert.Contains("-RuntimePackageKey $key -DryRun", script);
        Assert.Contains("Use Materialize-WindowsVendorRuntimeAssets.ps1 when local NVIDIA archives are available.", script);
        Assert.Contains("Resolve-RuntimeRoots.ps1", script);
    }

    [Fact]
    public void SplitRuntimeReadmeDocumentsVendorMaterializationBoundary()
    {
        string readme = ReadSource("pack", "runtime-split", "README.md");

        Assert.Contains("Materialize-WindowsVendorRuntimeAssets.ps1", readme);
        Assert.Contains("does not execute the cuDNN installer", readme);
        Assert.Contains("manifest-declared runtime DLL", readme);
        Assert.Contains("vendor-runtime-assets-summary.json", readme);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
