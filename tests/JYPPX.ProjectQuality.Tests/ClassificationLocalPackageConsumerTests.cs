using ClassificationSample;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ClassificationLocalPackageConsumerTests
{
    [Fact]
    public void CommandRunnerIsPublicAndOriginalEntryPointRemainsCompatible()
    {
        Assert.True(typeof(ClassificationCommand).IsPublic);
        Assert.NotNull(typeof(ClassificationCommand).GetMethod(nameof(ClassificationCommand.Run)));

        string entryPoint = ReadSource("samples", "ComputerVision", "01.Classification", "EntryPoint.cs");
        Assert.Contains("public static int Main(string[] args)", entryPoint, StringComparison.Ordinal);
        Assert.Contains("return ClassificationCommand.Run(args);", entryPoint, StringComparison.Ordinal);
    }

    [Fact]
    public void ClassificationIsANonPackablePublishedPackageConsumer()
    {
        string project = ReadSource("samples", "ComputerVision", "01.Classification", "Classification.csproj");
        string packages = ReadSource("build", "JYPPX.PublicSamplePackages.props");
        string openCvPackages = ReadSource("build", "JYPPX.OpenCvSamplePackages.props");

        Assert.Contains("<IsPackable>false</IsPackable>", project, StringComparison.Ordinal);
        Assert.DoesNotContain("<PackageId>", project, StringComparison.Ordinal);
        Assert.DoesNotContain("ProjectReference", project, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API", packages, StringComparison.Ordinal);
        Assert.Contains("JYPPX.OpenCV.CSharp.API", openCvPackages, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.TensorRT.CSharp.API.Classification", packages, StringComparison.Ordinal);
    }

    [Fact]
    public void RunnerPinsOfficialAssetsAndNeverPublishesOrBundlesVendorRuntimes()
    {
        string runner = ReadSource("eng", "Test-ClassificationLocalPackageConsumer.ps1");
        string workflow = ReadSource(".github", "workflows", "package-managed.yml");

        Assert.Contains("classification-resnet18-local-package-consumer-runtime", runner, StringComparison.Ordinal);
        Assert.Contains("43de394443f6fc3ccfd08cd9df61ee645ee5c51d1954c52267c221a438252f9e", runner, StringComparison.Ordinal);
        Assert.Contains("PackageReferenceCount=3 ProjectReferenceCount=0", runner, StringComparison.Ordinal);
        Assert.Contains("ControlledNegativeExitCode=", runner, StringComparison.Ordinal);
        Assert.Contains("vendorRuntimeEntryCount = $vendorEntries.Count", runner, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", runner, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release", runner, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Pack Classification managed extension", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.TensorRT.CSharp.API.Classification", workflow, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }

}
