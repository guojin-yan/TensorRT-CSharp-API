using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedSourceModuleLayoutTests
{
    public static TheoryData<string, string[]> ProjectModules => new()
    {
        {
            "JYPPX.CudaSharp",
            new[]
            {
                "Core", "Devices", "Diagnostics", "Drivers", "Events", "Graphs", "IPC", "Kernels",
                "Memory", "RuntimeCompilation", "Streams"
            }
        },
        {
            "JYPPX.TensorRtSharp",
            new[]
            {
                "Builder", "Callbacks/Core", "Callbacks/Debugging", "Callbacks/MemoryAllocation",
                "Callbacks/Monitoring", "ControlFlow", "Core", "Diagnostics", "Engine", "Execution",
                "Inference", "Layers", "Network", "Parsing", "Plugins", "Profiles", "Refit", "Runtime",
                "Serialization"
            }
        },
        {
            "JYPPX.TensorRtSharp.Tools",
            new[] { "Artifacts", "Build", "Core", "Runtime", "Trtexec" }
        }
    };

    [Theory]
    [MemberData(nameof(ProjectModules))]
    public void PublicApiFilesAreGroupedIntoNamedModules(string project, string[] modules)
    {
        string projectDirectory = Path.Combine(RepositoryPaths.Root, "src", project);

        Assert.Empty(Directory.EnumerateFiles(projectDirectory, "*.cs", SearchOption.TopDirectoryOnly));
        foreach (string module in modules)
        {
            string moduleDirectory = Path.Combine(
                new[] { projectDirectory }.Concat(module.Split('/')).ToArray());
            Assert.True(Directory.Exists(moduleDirectory), $"Missing managed source module: {project}/{module}");
            Assert.NotEmpty(Directory.EnumerateFiles(moduleDirectory, "*.cs", SearchOption.TopDirectoryOnly));
        }
    }

    [Fact]
    public void CudaDriverOwnersAreGroupedInDriversModule()
    {
        string cudaProjectDirectory = Path.Combine(RepositoryPaths.Root, "src", "JYPPX.CudaSharp");
        string driversDirectory = Path.Combine(cudaProjectDirectory, "Drivers");
        string[] driverFiles = Directory.EnumerateFiles(driversDirectory, "*.cs", SearchOption.TopDirectoryOnly)
            .Select(path => Path.GetFileName(path)!)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(
            new[] { "CudaDriver.cs", "CudaDriverKernelLaunch.cs", "CudaDriverModule.cs" },
            driverFiles);
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Core", "CudaDriver.cs")));
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Kernels", "CudaDriverKernelLaunch.cs")));
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Kernels", "CudaDriverModule.cs")));
    }
}
