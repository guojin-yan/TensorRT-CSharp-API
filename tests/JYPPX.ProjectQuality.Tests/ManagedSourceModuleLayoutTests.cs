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
        {
            "Builder",
            new[]
            {
                "NativeBridgeApi.BuilderBoundaryControls.cs",
                "NativeBridgeApi.BuilderBuildOutputs.cs",
                "NativeBridgeApi.BuilderConfigDiagnostics.cs",
                "NativeBridgeApi.BuilderConfigPluginSerialization.cs",
                "NativeBridgeApi.BuilderConfigRuntimeControls.cs",
                "NativeBridgeApi.TimingCacheLifecycle.cs",
                "NativeBridgeApi.Trt11TimingCache.cs"
            }
        },
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
        {
            "Engine",
            new[]
            {
                "NativeBridgeApi.Dims64EngineMetadata.cs",
                "NativeBridgeApi.EngineBoundaryControls.cs",
                "NativeBridgeApi.EngineDeploymentMetadata.cs",
                "NativeBridgeApi.EngineInspectorDiagnostics.cs",
                "NativeBridgeApi.EngineInspectorErrorRecorder.cs",
                "NativeBridgeApi.EngineProfileTensorValues.cs",
                "NativeBridgeApi.EngineRuntimeControls.cs"
            }
        },
        {
            "Execution",
            new[]
            {
                "NativeBridgeApi.Dims64ExecutionContext.cs",
                "NativeBridgeApi.ExecutionContextAddressAndAuxStreams.cs",
                "NativeBridgeApi.ExecutionContextBoundaryControls.cs",
                "NativeBridgeApi.ExecutionContextCreation.cs",
                "NativeBridgeApi.ExecutionContextDeploymentMetadata.cs",
                "NativeBridgeApi.ExecutionContextDiagnostics.cs",
                "NativeBridgeApi.ExecutionContextEngineMetadata.cs",
                "NativeBridgeApi.ExecutionContextRuntimeControls.cs"
            }
        },
        { "Inference", new[] { "NativeBridgeApi.SynchronousInference.cs" } },
        { "Interfaces", new[] { "NativeBridgeApi.OwnerScopedVersionedInterfaceMetadata.cs" } },
        {
            "Layers",
            new[]
            {
                "NativeBridgeApi.DeploymentLayerAttributes.cs",
                "NativeBridgeApi.Dims64LayerMetadata.cs",
                "NativeBridgeApi.LayerDeploymentMetadata.cs",
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
                "NativeBridgeApi.Dims64NetworkTensor.cs",
                "NativeBridgeApi.NetworkBoundaryControls.cs",
                "NativeBridgeApi.NetworkDiagnostics.cs",
                "NativeBridgeApi.Trt11SafeNetworkV2.cs"
            }
        },
        {
            "Parsing",
            new[]
            {
                "NativeBridgeApi.GlobalOnnxParserVersion.cs",
                "NativeBridgeApi.LegacyParserDiagnostics.cs",
                "NativeBridgeApi.OnnxConfig.cs",
                "NativeBridgeApi.OnnxModelBuffer.cs",
                "NativeBridgeApi.OnnxParserBuilderConfig.cs",
                "NativeBridgeApi.OnnxParserLayerOutputMetadata.cs",
                "NativeBridgeApi.OnnxParserSupport.cs",
                "NativeBridgeApi.OnnxWeightDescriptorParsing.cs",
                "NativeBridgeApi.ParserRefitterDiagnostics.cs"
            }
        },
        {
            "Plugins",
            new[]
            {
                "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs",
                "NativeBridgeApi.GlobalPluginRegistry.cs",
                "NativeBridgeApi.PluginInitialization.cs",
                "NativeBridgeApi.PluginLayerOwnerScopedQuerySnapshots.cs",
                "NativeBridgeApi.PluginRegistryInventory.cs",
                "NativeBridgeApi.PluginV2LayerMetadata.cs",
                "NativeBridgeApi.PluginV3LayerMetadata.cs",
                "NativeBridgeApi.RuntimePluginRegistryInventory.cs"
            }
        },
        {
            "Profiles",
            new[]
            {
                "NativeBridgeApi.Dims64OptimizationProfile.cs",
                "NativeBridgeApi.OptimizationProfileShapeValues.cs"
            }
        },
        {
            "Refit",
            new[]
            {
                "NativeBridgeApi.RefitterControls.cs",
                "NativeBridgeApi.RefitterDeploymentMetadata.cs"
            }
        },
        {
            "Runtime",
            new[]
            {
                "NativeBridgeApi.GlobalRuntimeVersion.cs",
                "NativeBridgeApi.RuntimeDeploymentControls.cs"
            }
        },
        {
            "Serialization",
            new[]
            {
                "NativeBridgeApi.EngineSerialization.cs",
                "NativeBridgeApi.HostMemoryMetadata.cs"
            }
        },
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

    [Fact]
    public void TensorRtDeploymentMetadataInteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] engineMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.EngineDeploymentMetadata.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextDeploymentMetadata.cs");
        string[] layerMethods = ReadInteropMethodNames(interopDirectory, "Layers", "NativeBridgeApi.LayerDeploymentMetadata.cs");
        string[] refitMethods = ReadInteropMethodNames(interopDirectory, "Refit", "NativeBridgeApi.RefitterDeploymentMetadata.cs");
        string[] sharedMethods = EnumeratePublicStaticMethodNames(File.ReadAllText(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.DeploymentMetadataShared.cs")));

        Assert.Equal(22, engineMethods.Length);
        Assert.All(
            engineMethods,
            method => Assert.True(
                method.Contains("Engine", StringComparison.Ordinal) || method == "CreateRefitter",
                $"Engine deployment interop contains a non-engine method: {method}"));
        Assert.Equal(28, executionMethods.Length);
        Assert.All(
            executionMethods,
            method => Assert.True(
                method.Contains("ExecutionContext", StringComparison.Ordinal) ||
                method.StartsWith("AllInput", StringComparison.Ordinal),
                $"Execution deployment interop contains a non-context method: {method}"));
        Assert.Equal(74, layerMethods.Length);
        Assert.DoesNotContain(layerMethods, method =>
            method.Contains("Engine", StringComparison.Ordinal) ||
            method.Contains("ExecutionContext", StringComparison.Ordinal) ||
            method.Contains("Refitter", StringComparison.Ordinal));
        Assert.Equal(6, refitMethods.Length);
        Assert.All(refitMethods, method => Assert.Contains("Refit", method, StringComparison.Ordinal));
        Assert.Empty(sharedMethods);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.DeploymentMetadata.cs")));
    }

    [Fact]
    public void TensorRtDims64InteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] networkMethods = ReadInteropMethodNames(interopDirectory, "Network", "NativeBridgeApi.Dims64NetworkTensor.cs");
        string[] engineMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.Dims64EngineMetadata.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.Dims64ExecutionContext.cs");
        string[] profileMethods = ReadInteropMethodNames(interopDirectory, "Profiles", "NativeBridgeApi.Dims64OptimizationProfile.cs");
        string[] layerMethods = ReadInteropMethodNames(interopDirectory, "Layers", "NativeBridgeApi.Dims64LayerMetadata.cs");

        Assert.Equal(6, networkMethods.Length);
        Assert.All(
            networkMethods,
            method => Assert.True(
                method.StartsWith("GetTensor", StringComparison.Ordinal) ||
                method.Contains("Network", StringComparison.Ordinal),
                $"Network Dims64 interop contains a non-network/tensor method: {method}"));
        Assert.Equal(4, engineMethods.Length);
        Assert.All(engineMethods, method => Assert.Contains("Engine", method, StringComparison.Ordinal));
        Assert.Equal(4, executionMethods.Length);
        Assert.All(executionMethods, method => Assert.Contains("ExecutionContext", method, StringComparison.Ordinal));
        Assert.Equal(2, profileMethods.Length);
        Assert.All(profileMethods, method => Assert.Contains("OptimizationProfile", method, StringComparison.Ordinal));
        Assert.Equal(27, layerMethods.Length);
        Assert.DoesNotContain(layerMethods, method =>
            method.Contains("Engine", StringComparison.Ordinal) ||
            method.Contains("ExecutionContext", StringComparison.Ordinal) ||
            method.Contains("Network", StringComparison.Ordinal) ||
            method.Contains("OptimizationProfile", StringComparison.Ordinal));
        Assert.All(
            networkMethods.Concat(engineMethods).Concat(executionMethods).Concat(profileMethods).Concat(layerMethods),
            method => Assert.EndsWith("64", method, StringComparison.Ordinal));
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11Dims64.cs")));
    }

    [Fact]
    public void TensorRtRuntimeControlsInteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] builderMethods = ReadInteropMethodNames(interopDirectory, "Builder", "NativeBridgeApi.BuilderConfigRuntimeControls.cs");
        string[] engineMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.EngineRuntimeControls.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextRuntimeControls.cs");

        Assert.Equal(19, builderMethods.Length);
        Assert.All(builderMethods, method => Assert.Contains("BuilderConfig", method, StringComparison.Ordinal));
        Assert.Equal(9, engineMethods.Length);
        Assert.All(engineMethods, method => Assert.Contains("Engine", method, StringComparison.Ordinal));
        Assert.Equal(4, executionMethods.Length);
        Assert.All(executionMethods, method => Assert.Contains("ExecutionContext", method, StringComparison.Ordinal));
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11RuntimeControls.cs")));
    }

    [Fact]
    public void TensorRtDiagnosticsInteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] builderMethods = ReadInteropMethodNames(interopDirectory, "Builder", "NativeBridgeApi.BuilderConfigDiagnostics.cs");
        string[] networkMethods = ReadInteropMethodNames(interopDirectory, "Network", "NativeBridgeApi.NetworkDiagnostics.cs");
        string[] engineMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.EngineInspectorDiagnostics.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextDiagnostics.cs");

        Assert.Equal(9, builderMethods.Length);
        Assert.All(builderMethods, method => Assert.Contains("BuilderConfig", method, StringComparison.Ordinal));
        Assert.Equal(7, networkMethods.Length);
        Assert.All(networkMethods, method => Assert.Contains("Network", method, StringComparison.Ordinal));
        Assert.Equal(4, engineMethods.Length);
        Assert.All(engineMethods, method => Assert.Contains("EngineInspector", method, StringComparison.Ordinal));
        Assert.Equal(15, executionMethods.Length);
        Assert.All(executionMethods, method => Assert.Contains("ExecutionContext", method, StringComparison.Ordinal));
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11Diagnostics.cs")));
    }

    [Fact]
    public void TensorRtGlobalRuntimePluginProbeInteropIsSplitByBehavior()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] runtimeMethods = ReadInteropMethodNames(interopDirectory, "Runtime", "NativeBridgeApi.GlobalRuntimeVersion.cs");
        string[] parsingMethods = ReadInteropMethodNames(interopDirectory, "Parsing", "NativeBridgeApi.GlobalOnnxParserVersion.cs");
        string[] pluginMethods = ReadInteropMethodNames(interopDirectory, "Plugins", "NativeBridgeApi.GlobalPluginRegistry.cs");
        string sharedSource = File.ReadAllText(Path.Combine(interopDirectory, "NativeBridgeApi.GlobalProbeShared.cs"));

        Assert.Equal(7, runtimeMethods.Length);
        Assert.DoesNotContain(runtimeMethods, method => method.Contains("Plugin", StringComparison.Ordinal));
        Assert.Equal(new[] { "GetGlobalOnnxParserVersion" }, parsingMethods);
        Assert.Equal(8, pluginMethods.Length);
        Assert.All(pluginMethods, method => Assert.Contains("GlobalPlugin", method, StringComparison.Ordinal));
        Assert.Empty(EnumeratePublicStaticMethodNames(sharedSource));
        Assert.Contains("private static BridgeProbeException UnsupportedGlobalRuntimeProbeLine()", sharedSource, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.GlobalRuntimePluginProbe.cs")));
    }

    [Fact]
    public void TensorRtSafeDeferredUpliftInteropIsSplitByBehavior()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] pluginMethods = ReadInteropMethodNames(interopDirectory, "Plugins", "NativeBridgeApi.PluginInitialization.cs");
        string[] parsingMethods = ReadInteropMethodNames(interopDirectory, "Parsing", "NativeBridgeApi.OnnxWeightDescriptorParsing.cs");

        Assert.Equal(new[] { "InitializeLibNvInferPlugins" }, pluginMethods);
        Assert.Equal(new[] { "ParseOnnxWithWeightDescriptors" }, parsingMethods);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.SafeDeferredUplift.cs")));
    }

    [Fact]
    public void TensorRtBoundaryControlsInteropIsSplitByOwner()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string[] builderMethods = ReadInteropMethodNames(interopDirectory, "Builder", "NativeBridgeApi.BuilderBoundaryControls.cs");
        string[] engineMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.EngineBoundaryControls.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextBoundaryControls.cs");
        string[] networkMethods = ReadInteropMethodNames(interopDirectory, "Network", "NativeBridgeApi.NetworkBoundaryControls.cs");
        string sharedSource = File.ReadAllText(Path.Combine(interopDirectory, "NativeBridgeApi.OwnerErrorRecorderSnapshotShared.cs"));

        Assert.Equal(12, builderMethods.Length);
        Assert.All(builderMethods, method => Assert.True(
            method.Contains("Builder", StringComparison.Ordinal) || method == "IsNetworkSupported",
            $"Builder boundary interop contains a non-builder method: {method}"));
        Assert.Equal(5, engineMethods.Length);
        Assert.All(engineMethods, method => Assert.Contains("Engine", method, StringComparison.Ordinal));
        Assert.Equal(3, executionMethods.Length);
        Assert.All(executionMethods, method => Assert.Contains("ExecutionContext", method, StringComparison.Ordinal));
        Assert.Equal(5, networkMethods.Length);
        Assert.All(networkMethods, method => Assert.True(
            method.Contains("Network", StringComparison.Ordinal) || method == "AddTopKV2Layer",
            $"Network boundary interop contains a non-network method: {method}"));
        Assert.Empty(EnumeratePublicStaticMethodNames(sharedSource));
        Assert.Contains("private static TensorRtErrorRecorderSnapshot ReadOwnerErrorRecorderSnapshot(", sharedSource, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11BoundaryControls.cs")));
    }

    [Fact]
    public void TensorRtFourteenthBatchInteropIsSplitByOwnerAndFeature()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string builderBuildSource = File.ReadAllText(Path.Combine(interopDirectory, "Builder", "NativeBridgeApi.BuilderBuildOutputs.cs"));
        string[] builderBuildMethods = EnumeratePublicStaticMethodNames(builderBuildSource);
        string[] builderConfigMethods = ReadInteropMethodNames(interopDirectory, "Builder", "NativeBridgeApi.BuilderConfigPluginSerialization.cs");
        string[] hostMemoryMethods = ReadInteropMethodNames(interopDirectory, "Serialization", "NativeBridgeApi.HostMemoryMetadata.cs");
        string[] profileMethods = ReadInteropMethodNames(interopDirectory, "Profiles", "NativeBridgeApi.OptimizationProfileShapeValues.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextAddressAndAuxStreams.cs");

        Assert.Equal(new[] { "BuildEngineWithConfig", "BuildSerializedNetworkWithKernelText" }, builderBuildMethods);
        Assert.Contains("internal readonly struct NativeTensorRtSerializedNetworkWithKernelText", builderBuildSource, StringComparison.Ordinal);
        Assert.Equal(new[] { "ClearBuilderConfigFlag", "SetBuilderConfigPluginsToSerialize" }, builderConfigMethods);
        Assert.Equal(new[] { "GetHostMemoryDataType" }, hostMemoryMethods);
        Assert.Equal(
            new[] { "SetOptimizationProfileShapeValuesV2", "GetOptimizationProfileShapeValueCountV2", "GetOptimizationProfileShapeValuesV2" },
            profileMethods);
        Assert.Equal(
            new[]
            {
                "ClearExecutionContextTensorAddress",
                "ClearExecutionContextInputTensorAddress",
                "ClearExecutionContextOutputTensorAddress",
                "ClearExecutionContextDeviceMemory",
                "ClearExecutionContextInputConsumedEvent",
                "SetExecutionContextAuxStreams"
            },
            executionMethods);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11FourteenthBatch.cs")));
    }

    [Fact]
    public void TensorRtFifteenthBatchInteropIsSplitByOwnerAndFeature()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string engineProfileSource = File.ReadAllText(Path.Combine(interopDirectory, "Engine", "NativeBridgeApi.EngineProfileTensorValues.cs"));
        string[] engineProfileMethods = EnumeratePublicStaticMethodNames(engineProfileSource);
        string[] inspectorMethods = ReadInteropMethodNames(interopDirectory, "Engine", "NativeBridgeApi.EngineInspectorErrorRecorder.cs");
        string[] executionMethods = ReadInteropMethodNames(interopDirectory, "Execution", "NativeBridgeApi.ExecutionContextEngineMetadata.cs");

        Assert.Equal(new[] { "GetEngineProfileTensorValues", "GetEngineProfileTensorValuesV2" }, engineProfileMethods);
        Assert.Contains("private static void ValidateProfileTensorValuesInput(", engineProfileSource, StringComparison.Ordinal);
        Assert.Contains("private static void EnsureTensorRt10(", engineProfileSource, StringComparison.Ordinal);
        Assert.Equal(new[] { "ClearEngineInspectorErrorRecorder" }, inspectorMethods);
        Assert.Equal(
            new[]
            {
                "GetExecutionContextInputConsumedEventAddressValue",
                "GetExecutionContextRuntimeConfigAllocationStrategy",
                "GetExecutionContextEngineName",
                "GetExecutionContextEngineIOTensorCount",
                "GetExecutionContextEngineLayerCount",
                "GetExecutionContextEngineOptimizationProfileCount"
            },
            executionMethods);
        Assert.False(File.Exists(Path.Combine(
            interopDirectory,
            "NativeBridgeApi.Trt11FifteenthBatch.cs")));
    }

    [Fact]
    public void TensorRtRootTimingCacheLifecycleIsSplitIntoBuilderModule()
    {
        string interopDirectory = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop");
        string rootSource = File.ReadAllText(Path.Combine(interopDirectory, "NativeBridgeApi.cs"));
        string[] rootMethods = EnumeratePublicStaticMethodNames(rootSource);
        string[] timingCacheMethods = ReadInteropMethodNames(
            interopDirectory,
            "Builder",
            "NativeBridgeApi.TimingCacheLifecycle.cs");

        Assert.Equal(
            new[] { "CreateTimingCache", "SetTimingCache", "SerializeTimingCache" },
            timingCacheMethods);
        Assert.DoesNotContain("CreateTimingCache", rootMethods);
        Assert.DoesNotContain("SetTimingCache", rootMethods);
        Assert.DoesNotContain("SerializeTimingCache", rootMethods);
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
