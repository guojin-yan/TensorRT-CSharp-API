using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class VendorRuntimeAssetMaterializationTests
{
    [Fact]
    public void VendorMaterializerAndFullRuntimeProjectsAreRemoved()
    {
        Assert.False(File.Exists(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Materialize-WindowsVendorRuntimeAssets.ps1")));

        Assert.Empty(Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "pack", "runtime"),
            "*.csproj",
            SearchOption.AllDirectories));
    }

    [Fact]
    public void SplitRuntimeProjectsContainOnlyProjectOwnedBridges()
    {
        string[] projects = Directory.GetFiles(
            Path.Combine(RepositoryPaths.Root, "pack", "runtime-split"),
            "*.csproj",
            SearchOption.AllDirectories);

        Assert.NotEmpty(projects);
        Assert.All(projects, static projectPath =>
        {
            Assert.EndsWith(".Bridge.csproj", projectPath, StringComparison.Ordinal);
            XDocument document = XDocument.Load(projectPath);
            string xml = document.ToString(SaveOptions.DisableFormatting);
            Assert.Contains("<JYPPXPackageKind>bridge</JYPPXPackageKind>", xml, StringComparison.Ordinal);
            string[] includes = document.Descendants()
                .Select(static element => element.Attribute("Include")?.Value)
                .Where(static value => !string.IsNullOrWhiteSpace(value))
                .Cast<string>()
                .ToArray();
            Assert.DoesNotContain(includes, static value => value.Contains("nvinfer", StringComparison.OrdinalIgnoreCase));
            Assert.DoesNotContain(includes, static value => value.Contains("cudnn", StringComparison.OrdinalIgnoreCase));
            Assert.DoesNotContain(includes, static value => value.Contains("nvrtc", StringComparison.OrdinalIgnoreCase));
        });
    }

    [Fact]
    public void RuntimeDocumentationRequiresConsumerInstalledNvidiaDependencies()
    {
        string runtimeReadme = ReadSource("pack", "runtime", "README.md");
        string splitReadme = ReadSource("pack", "runtime-split", "README.md");

        Assert.Contains("Consumers install matching NVIDIA dependencies themselves", runtimeReadme, StringComparison.Ordinal);
        Assert.Contains("never included", splitReadme, StringComparison.Ordinal);
        Assert.Contains("Only project-owned native bridge packages are active", splitReadme, StringComparison.Ordinal);
        Assert.DoesNotContain("Materialize-WindowsVendorRuntimeAssets.ps1", runtimeReadme, StringComparison.Ordinal);
        Assert.DoesNotContain("Materialize-WindowsVendorRuntimeAssets.ps1", splitReadme, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
