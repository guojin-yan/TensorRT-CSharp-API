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

        string entryPoint = ReadSource("samples", "Classification", "EntryPoint.cs");
        Assert.Contains("public static int Main(string[] args)", entryPoint, StringComparison.Ordinal);
        Assert.Contains("return ClassificationCommand.Run(args);", entryPoint, StringComparison.Ordinal);
    }

    [Fact]
    public void ClassificationPackageAndConsumerKeepTheThreePackageBoundary()
    {
        string project = ReadSource("samples", "Classification", "Classification.csproj");
        string template = ReadSource("samples", "Classification.PackageConsumer", "Classification.PackageConsumer.csproj.template");
        string program = ReadSource("samples", "Classification.PackageConsumer", "Program.cs");

        Assert.Contains("<PackageId>JYPPX.TensorRT.CSharp.API.Classification</PackageId>", project, StringComparison.Ordinal);
        Assert.Contains("<IsPackable>true</IsPackable>", project, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.CudaSharp.csproj", project, StringComparison.Ordinal);
        Assert.Equal(3, CountOccurrences(template, "<PackageReference"));
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("<Reference ", template, StringComparison.Ordinal);
        Assert.Contains("ClassificationCommand.Run(args)", program, StringComparison.Ordinal);
        Assert.Contains("ClassificationPackageConsumer ProjectReference=False", program, StringComparison.Ordinal);
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
        Assert.Contains("Pack Classification managed extension", workflow, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API.Classification", workflow, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }

    private static int CountOccurrences(string text, string value)
    {
        int count = 0;
        int offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
