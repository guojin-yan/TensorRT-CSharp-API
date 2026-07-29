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
                "Inference", "Interfaces", "Layers", "Network", "Parsing", "Plugins", "Profiles", "Refit",
                "Runtime", "Serialization", "Weights"
            }
        },
        {
            "JYPPX.TensorRtSharp.Tools",
            new[] { "Artifacts", "Build", "Core", "Refit", "Runtime", "Trtexec" }
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
        string[] driverFiles = EnumerateModuleFiles(cudaProjectDirectory, "Drivers");

        Assert.Equal(
            new[] { "CudaDriver.cs", "CudaDriverKernelLaunch.cs", "CudaDriverModule.cs" },
            driverFiles);
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Core", "CudaDriver.cs")));
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Kernels", "CudaDriverKernelLaunch.cs")));
        Assert.False(File.Exists(Path.Combine(cudaProjectDirectory, "Kernels", "CudaDriverModule.cs")));
    }

    [Fact]
    public void TensorRtInterfacesAndWeightsAreGroupedByResponsibility()
    {
        string tensorRtProjectDirectory = Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp");
        string[] interfaceFiles =
        {
            "TensorRtInterfaceInfo.cs",
            "TensorRtOwnerScopedVersionedInterfaceMetadata.cs",
            "TensorRtVersionedInterfaceMetadata.cs"
        };
        string[] weightsFiles =
        {
            "TensorRtWeights.cs",
            "TensorRtWeightsInfo.cs",
            "TensorRtWeightsRole.cs"
        };

        Assert.Equal(interfaceFiles, EnumerateModuleFiles(tensorRtProjectDirectory, "Interfaces"));
        Assert.Equal(weightsFiles, EnumerateModuleFiles(tensorRtProjectDirectory, "Weights"));

        string coreDirectory = Path.Combine(tensorRtProjectDirectory, "Core");
        Assert.All(interfaceFiles, file => Assert.False(File.Exists(Path.Combine(coreDirectory, file))));
        Assert.All(weightsFiles, file => Assert.False(File.Exists(Path.Combine(coreDirectory, file))));
    }

    [Fact]
    public void TensorRtToolsRefitSnapshotsHaveDedicatedModule()
    {
        string toolsProjectDirectory = Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools");
        string[] refitFiles =
        {
            "OnnxEngineRefitPersistenceSnapshot.cs",
            "OnnxEngineRefitSnapshot.cs"
        };

        Assert.Equal(refitFiles, EnumerateModuleFiles(toolsProjectDirectory, "Refit"));

        string buildDirectory = Path.Combine(toolsProjectDirectory, "Build");
        Assert.All(refitFiles, file => Assert.False(File.Exists(Path.Combine(buildDirectory, file))));
    }

    private static string[] EnumerateModuleFiles(string projectDirectory, string module)
    {
        return Directory.EnumerateFiles(
                Path.Combine(projectDirectory, module),
                "*.cs",
                SearchOption.TopDirectoryOnly)
            .Select(path => Path.GetFileName(path)!)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
    }
}
