using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedPackageTests
{
    [Fact]
    public void ManagedPackageProjectCollectsManagedAssemblies()
    {
        string projectPath = Path.Combine(RepositoryPaths.Root, "pack", "JYPPX.TensorRT.CSharp.API", "JYPPX.TensorRT.CSharp.API.csproj");
        XDocument project = XDocument.Load(projectPath);
        string xml = project.ToString(SaveOptions.DisableFormatting);

        Assert.Contains("BuildManagedAssembliesForPackage", xml);
        Assert.Contains("JYPPX.TensorRtSharp.dll", xml);
        Assert.Contains("JYPPX.CudaSharp.dll", xml);
        Assert.Contains("JYPPX.Shared.dll", xml);
        Assert.DoesNotContain("<SuppressDependenciesWhenPacking>true</SuppressDependenciesWhenPacking>", xml);
    }
}
