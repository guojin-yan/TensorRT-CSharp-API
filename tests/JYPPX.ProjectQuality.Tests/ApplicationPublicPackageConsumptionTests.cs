using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ApplicationPublicPackageConsumptionTests
{
    [Theory]
    [InlineData("OnnxToEngine", "OnnxToEngine.csproj")]
    [InlineData("TensorRtExec", "TensorRtExec.csproj")]
    public void CompleteApplicationsConsumeThePublishedManagedPackage(
        string applicationDirectory,
        string projectFileName)
    {
        string project = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            applicationDirectory,
            projectFileName));
        string normalized = project.Replace('/', '\\');

        Assert.Contains("JYPPX.PublicSamplePackages.props", project, StringComparison.Ordinal);
        Assert.Contains(
            "_shared\\JYPPX.TensorRtSharp.ApplicationTools\\JYPPX.TensorRtSharp.ApplicationTools.csproj",
            normalized,
            StringComparison.Ordinal);
        Assert.DoesNotContain("src\\JYPPX.CudaSharp", normalized, StringComparison.Ordinal);
        Assert.DoesNotContain("src\\JYPPX.TensorRtSharp\\", normalized, StringComparison.Ordinal);
    }

    [Fact]
    public void ApplicationToolsCompileLinkedSourceAgainstThePublicPackageOnly()
    {
        string project = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "_shared",
            "JYPPX.TensorRtSharp.ApplicationTools",
            "JYPPX.TensorRtSharp.ApplicationTools.csproj"));

        Assert.Contains("JYPPX.PublicSamplePackages.props", project, StringComparison.Ordinal);
        Assert.Contains("src\\JYPPX.TensorRtSharp.Tools\\**\\*.cs", project, StringComparison.Ordinal);
        Assert.Contains("<IsPackable>false</IsPackable>", project, StringComparison.Ordinal);
        Assert.DoesNotContain("<ProjectReference", project, StringComparison.Ordinal);
        Assert.DoesNotContain("src\\JYPPX.CudaSharp", project, StringComparison.Ordinal);
        Assert.DoesNotContain("src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj", project, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedPackageRulesUseMaintainedTensorRtLineAndPinnedOpenCvRelease()
    {
        string tensorRtProps = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "build",
            "JYPPX.PublicSamplePackages.props"));
        string openCvProps = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "build",
            "JYPPX.OpenCvSamplePackages.props"));
        string sampleReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));
        string applicationReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "README.md"));
        string combinedReadmes = sampleReadme + applicationReadme;

        Assert.Contains("4.0.0-*", tensorRtProps, StringComparison.Ordinal);
        Assert.Contains(">5.0.0<", openCvProps, StringComparison.Ordinal);
        Assert.Contains(
            "dotnet add package JYPPX.TensorRT.CSharp.API --version \"4.0.0-*\"",
            combinedReadmes,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "dotnet add package JYPPX.TensorRT.CSharp.API --prerelease",
            combinedReadmes,
            StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet add package JYPPX.TensorRT.CSharp.API --version \"4.0.0-preview.1\"", combinedReadmes, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dotnet add package JYPPX.TensorRT.CSharp.API --version \"4.0.0\"", combinedReadmes, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("4.0.0-preview.1", combinedReadmes, StringComparison.Ordinal);
    }
}
