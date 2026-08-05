using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class NuGetPackageBrandingTests
{
    [Fact]
    public void CanonicalLogoIsAValidTrackedJpegAsset()
    {
        string path = Path.Combine(RepositoryPaths.Root, "nuget", "logo.jpg");
        byte[] bytes = File.ReadAllBytes(path);

        Assert.True(bytes.Length > 4_096);
        Assert.Equal(0xFF, bytes[0]);
        Assert.Equal(0xD8, bytes[1]);
        Assert.Equal(0xFF, bytes[^2]);
        Assert.Equal(0xD9, bytes[^1]);
    }

    [Theory]
    [InlineData("pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj", "../../README.md", "../../nuget/logo.jpg")]
    [InlineData("samples/Classification/Classification.csproj", "../../README.md", "../../nuget/logo.jpg")]
    [InlineData("samples/YoloVision/YoloVision.csproj", "../../README.md", "../../nuget/logo.jpg")]
    public void ManagedPackagesUseRootEnglishReadmeAndCanonicalLogo(
        string projectPath,
        string expectedReadme,
        string expectedLogo)
    {
        XDocument project = XDocument.Load(Path.Combine(RepositoryPaths.Root, projectPath));
        AssertPackageMetadata(project, expectedReadme, expectedLogo);
    }

    [Theory]
    [InlineData("pack/runtime/Directory.Build.props")]
    [InlineData("pack/runtime-split/Directory.Build.props")]
    public void NativePackageFamiliesUseRootEnglishReadmeAndCanonicalLogo(string propsPath)
    {
        XDocument project = XDocument.Load(Path.Combine(RepositoryPaths.Root, propsPath));
        AssertPackageMetadata(project, "../../README.md", "../../nuget/logo.jpg");
    }

    private static void AssertPackageMetadata(XDocument project, string expectedReadme, string expectedLogo)
    {
        Assert.Contains(
            project.Descendants("PackageReadmeFile"),
            static element => element.Value == "README.md");
        Assert.Contains(
            project.Descendants("PackageIcon"),
            static element => element.Value == "logo.jpg");

        XElement readme = Assert.Single(project.Descendants("None").Where(
            static element => (string?)element.Attribute("Link") == "README.md"));
        XElement logo = Assert.Single(project.Descendants("None").Where(
            static element => (string?)element.Attribute("Link") == "logo.jpg"));

        Assert.Equal(expectedReadme, NormalizePath((string?)readme.Attribute("Include")));
        Assert.Equal(expectedLogo, NormalizePath((string?)logo.Attribute("Include")));
        Assert.Equal("true", (string?)readme.Attribute("Pack"));
        Assert.Equal("true", (string?)logo.Attribute("Pack"));
        Assert.Equal("\\", (string?)readme.Attribute("PackagePath"));
        Assert.Equal("\\", (string?)logo.Attribute("PackagePath"));
    }

    private static string NormalizePath(string? path) => path!.Replace('\\', '/');
}
