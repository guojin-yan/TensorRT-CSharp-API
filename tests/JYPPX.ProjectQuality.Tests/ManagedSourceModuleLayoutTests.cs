using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedSourceModuleLayoutTests
{
    public static TheoryData<string, string[]> CudaInteropFeatureModules => new()
    {
        { "Devices", new[] { "NativeCudaApi.DeviceResources.cs", "NativeCudaApi.ExecutionContext.cs" } },
        { "Diagnostics", new[] { "NativeCudaApi.Logs.cs" } },
        { "Drivers", new[] { "NativeCudaApi.Driver.cs" } },
        { "IPC", new[] { "NativeCudaApi.IpcExports.cs", "NativeCudaApi.IpcImports.cs" } },
        { "Kernels", new[] { "NativeCudaApi.KernelLibrary.cs" } },
        { "RuntimeCompilation", new[] { "NativeCudaApi.Rtc.cs" } }
    };

    public static TheoryData<string, string[]> TensorRtInteropFeatureModules => new()
    {
        { "Builder", new[] { "NativeBridgeApi.Trt11TimingCache.cs" } },
        {
            "Callbacks",
            new[]
            {
                "NativeBridgeApi.AllocatorOwnerDryRun.cs",
                "NativeBridgeApi.CallbackInterfaceInfo.cs",
                "NativeBridgeApi.ExecutionContextCallbackState.cs",
                "TensorRtLoggerCallback.cs",
                "TensorRtProfilerCallback.cs",
                "TensorRtProgressMonitorCallback.cs"
            }
        },
        { "ControlFlow", new[] { "NativeBridgeApi.Trt11ControlFlow.cs" } },
        {
            "Diagnostics",
            new[]
            {
                "NativeBridgeApi.ErrorCodeMetadata.cs",
                "NativeBridgeApi.Trt11BuildProbe.cs"
            }
        },
        { "Execution", new[] { "NativeBridgeApi.ExecutionContextCreation.cs" } },
        { "Inference", new[] { "NativeBridgeApi.SynchronousInference.cs" } },
        { "Interfaces", new[] { "NativeBridgeApi.OwnerScopedVersionedInterfaceMetadata.cs" } },
        {
            "Layers",
            new[]
            {
                "NativeBridgeApi.DeploymentLayerAttributes.cs",
                "NativeBridgeApi.Quantization.cs",
                "NativeBridgeApi.ThirtyThirdBatchLayerAttributes.cs",
                "NativeBridgeApi.Trt11Attention.cs",
                "NativeBridgeApi.Trt11FillInt64.cs",
                "NativeBridgeApi.Trt11LayerTensorMetadata.cs",
                "NativeBridgeApi.Trt11TransformerMetadata.cs",
                "NativeBridgeApi.Trt8RnnV2Diagnostics.cs"
            }
        },
        {
            "Network",
            new[]
            {
                "NativeBridgeApi.DeploymentNetworkLayers.cs",
                "NativeBridgeApi.Trt11SafeNetworkV2.cs"
            }
        },
        {
            "Parsing",
            new[]
            {
                "NativeBridgeApi.LegacyParserDiagnostics.cs",
                "NativeBridgeApi.OnnxConfig.cs",
                "NativeBridgeApi.OnnxModelBuffer.cs",
                "NativeBridgeApi.OnnxParserBuilderConfig.cs",
                "NativeBridgeApi.OnnxParserLayerOutputMetadata.cs",
                "NativeBridgeApi.OnnxParserSupport.cs",
                "NativeBridgeApi.ParserRefitterDiagnostics.cs"
            }
        },
        {
            "Plugins",
            new[]
            {
                "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs",
                "NativeBridgeApi.PluginLayerOwnerScopedQuerySnapshots.cs",
                "NativeBridgeApi.PluginRegistryInventory.cs",
                "NativeBridgeApi.PluginV2LayerMetadata.cs",
                "NativeBridgeApi.PluginV3LayerMetadata.cs",
                "NativeBridgeApi.RuntimePluginRegistryInventory.cs"
            }
        },
        { "Refit", new[] { "NativeBridgeApi.RefitterControls.cs" } },
        { "Runtime", new[] { "NativeBridgeApi.RuntimeDeploymentControls.cs" } },
        { "Serialization", new[] { "NativeBridgeApi.EngineSerialization.cs" } },
        { "Weights", new[] { "NativeBridgeApi.LayerWeightsInfo.cs" } }
    };

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

    [Theory]
    [MemberData(nameof(CudaInteropFeatureModules))]
    public void CudaInteropFilesAreGroupedIntoFeatureModules(string module, string[] expectedFiles)
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.CudaSharp",
            "Internal",
            "Interop");

        Assert.Equal(expectedFiles, EnumerateModuleFiles(interopDirectory, module));
        Assert.All(expectedFiles, file => Assert.False(File.Exists(Path.Combine(interopDirectory, file))));
    }

    [Theory]
    [MemberData(nameof(TensorRtInteropFeatureModules))]
    public void TensorRtInteropFilesAreGroupedIntoFeatureModules(string module, string[] expectedFiles)
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");

        Assert.Equal(expectedFiles, EnumerateModuleFiles(interopDirectory, module));
        Assert.All(expectedFiles, file => Assert.False(File.Exists(Path.Combine(interopDirectory, file))));
    }

    [Fact]
    public void TensorRtDeploymentInteropIsSplitByNetworkAndLayerOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string networkSource = File.ReadAllText(Path.Combine(
            interopDirectory,
            "Network",
            "NativeBridgeApi.DeploymentNetworkLayers.cs"));
        string layerSource = File.ReadAllText(Path.Combine(
            interopDirectory,
            "Layers",
            "NativeBridgeApi.DeploymentLayerAttributes.cs"));

        string[] networkMethods = EnumeratePublicStaticMethodNames(networkSource);
        string[] layerMethods = EnumeratePublicStaticMethodNames(layerSource);

        Assert.Equal(20, networkMethods.Length);
        Assert.All(
            networkMethods,
            method => Assert.True(
                method.StartsWith("Add", StringComparison.Ordinal) ||
                method is "MarkWeightsRefittable" or "UnmarkWeightsRefittable" or
                    "AreWeightsMarkedRefittable" or "SetWeightsName",
                $"Network interop contains a layer-attribute method: {method}"));
        Assert.Equal(66, layerMethods.Length);
        Assert.DoesNotContain(layerMethods, method => method.StartsWith("Add", StringComparison.Ordinal));
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11DeploymentAdditions.cs")));
    }

    [Fact]
    public void TensorRtRuntimeSerializationRefitInteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] runtimeMethods = ReadInteropMethodNames(interopDirectory, "Runtime", "NativeBridgeApi.RuntimeDeploymentControls.cs");
        string[] serializationMethods = ReadInteropMethodNames(interopDirectory, "Serialization", "NativeBridgeApi.EngineSerialization.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextCreation.cs");
        string[] refitMethods = ReadInteropMethodNames(interopDirectory, "Refit", "NativeBridgeApi.RefitterControls.cs");

        Assert.Equal(17, runtimeMethods.Length);
        Assert.All(runtimeMethods, method => Assert.Contains("Runtime", method, StringComparison.Ordinal));
        Assert.Equal(8, serializationMethods.Length);
        Assert.All(
            serializationMethods,
            method => Assert.True(
                method.StartsWith("Serialize", StringComparison.Ordinal) ||
                method.Contains("Serialization", StringComparison.Ordinal),
                $"Serialization interop contains a non-serialization method: {method}"));
        Assert.Equal(
            new[]
            {
                "CreateRuntimeConfig",
                "CreateExecutionContext",
                "CreateExecutionContext",
                "SetRuntimeConfigAllocationStrategy",
                "GetRuntimeConfigAllocationStrategy"
            },
            executionMethods);
        Assert.Equal(23, refitMethods.Length);
        Assert.All(refitMethods, method => Assert.Contains("Refit", method, StringComparison.Ordinal));
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs")));
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

    private static string[] EnumeratePublicStaticMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+static\s+[^\s]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] ReadInteropMethodNames(
        string interopDirectory,
        string module,
        string fileName)
    {
        return EnumeratePublicStaticMethodNames(File.ReadAllText(Path.Combine(
            interopDirectory,
            module,
            fileName)));
    }
}
