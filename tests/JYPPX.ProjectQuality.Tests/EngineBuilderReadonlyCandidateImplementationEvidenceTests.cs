using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class EngineBuilderReadonlyCandidateImplementationEvidenceTests
{
    [Fact]
    public void EngineAndBuilderReadonlyCandidatesAreLinkedToPointerFreeWrappers()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        string candidateText = File.ReadAllText(candidatePath);
        using JsonDocument document = JsonDocument.Parse(candidateText);

        JsonElement groups = document.RootElement.GetProperty("groups");
        JsonElement engineLayerCandidate = FindCandidate(groups, "engine-layer-metadata-001");
        JsonElement engineTensorCandidate = FindCandidate(groups, "engine-tensor-binding-002");
        JsonElement builderConfigCandidate = FindCandidate(groups, "builder-config-readback-001");

        AssertCandidateImplemented(engineLayerCandidate);
        AssertCandidateImplemented(engineTensorCandidate);
        AssertCandidateImplemented(builderConfigCandidate);

        AssertEvidenceContains(engineLayerCandidate, "publicSurface", "TensorRtEngineInspector.GetLayerInformation");
        AssertEvidenceContains(engineLayerCandidate, "publicSurface", "TensorRtLayer.GetInputTensorMetadata");
        AssertEvidenceContains(engineTensorCandidate, "publicSurface", "TensorRtEngine.GetTensorBinding");
        AssertEvidenceContains(engineTensorCandidate, "publicSurface", "TensorRtEngine.GetDeploymentSnapshot");
        AssertEvidenceContains(builderConfigCandidate, "publicSurface", "TensorRtBuilderConfig.GetDeploymentSnapshot");
        AssertEvidenceContains(builderConfigCandidate, "publicSurface", "TensorRtBuilderConfig.SerializedPluginPathCountCompatibility");

        string engineApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.BindingReports.cs");
        string engineProfileApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11FifteenthBatch.cs");
        string engineBindingReport = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineBindingReport.cs");
        string engineTensorBinding = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineTensorBinding.cs");
        string engineDeploymentSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineDeploymentSnapshot.cs");
        string engineInspector = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineInspector.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngineInspector.Trt11Diagnostics.cs");
        string layerTensorMetadata = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayer.Trt11LayerTensorMetadata.cs");
        string layerTensorMetadataModel = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayerTensorMetadata.cs");
        string builderConfigApi = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11Diagnostics.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11DeploymentSnapshot.cs");
        string builderConfigSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfigDeploymentSnapshot.cs");
        string builderConfigSerializedPluginSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfigSerializedPluginSnapshot.cs");

        Assert.Contains("public TensorRtEngineTensorBinding GetTensorBinding", engineApi);
        Assert.Contains("public TensorRtEngineBindingReport GetBindingReport", engineApi);
        Assert.Contains("public TensorRtEngineDeploymentSnapshot GetDeploymentSnapshot", engineProfileApi);
        Assert.Contains("public TensorRtEngineProfileTensorValuesSnapshot GetProfileTensorValuesSnapshot", engineProfileApi);
        Assert.Contains("public sealed class TensorRtEngineBindingReport", engineBindingReport);
        Assert.Contains("public sealed class TensorRtEngineTensorBinding", engineTensorBinding);
        Assert.Contains("public sealed class TensorRtEngineDeploymentSnapshot", engineDeploymentSnapshot);
        Assert.Contains("public TensorRtEngineDeploymentSummary ToSummary()", engineDeploymentSnapshot);
        Assert.Contains("public sealed class TensorRtEngineDeploymentSummary", engineDeploymentSnapshot);
        Assert.Contains("public bool CopiedTensorCountMatchesReportedIOTensorCount", engineDeploymentSnapshot);
        Assert.Contains("public bool PointerFreeCopiedSummary", engineDeploymentSnapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof", engineDeploymentSnapshot);
        Assert.Contains("public bool CanDeleteDeferredRecord", engineDeploymentSnapshot);
        Assert.Contains("GetLayerInformation", engineInspector);
        Assert.Contains("public TensorRtLayerTensorMetadata GetInputTensorMetadata", layerTensorMetadata);
        Assert.Contains("public TensorRtLayerTensorMetadata GetOutputTensorMetadata", layerTensorMetadata);
        Assert.Contains("public sealed class TensorRtLayerTensorMetadata", layerTensorMetadataModel);

        Assert.Contains("public TensorRtBuilderFlags GetFlags", builderConfigApi);
        Assert.Contains("public TensorRtDeviceType GetDefaultDeviceType", builderConfigApi);
        Assert.Contains("public ulong GetMemoryPoolLimit", builderConfigApi);
        Assert.Contains("public TensorRtQuantizationFlags GetQuantizationFlags", builderConfigApi);
        Assert.Contains("public int SerializedPluginPathCountCompatibility", builderConfigApi);
        Assert.Contains("public TensorRtBuilderConfigDeploymentSnapshot GetDeploymentSnapshot", builderConfigApi);
        Assert.Contains("public sealed class TensorRtBuilderConfigDeploymentSnapshot", builderConfigSnapshot);
        Assert.Contains("public TensorRtBuilderConfigDeploymentSummary ToSummary()", builderConfigSnapshot);
        Assert.Contains("public sealed class TensorRtBuilderConfigDeploymentSummary", builderConfigSnapshot);
        Assert.Contains("public bool CopiedPluginCountMatchesReportedCount", builderConfigSnapshot);
        Assert.Contains("public bool PointerFreeCopiedSummary", builderConfigSnapshot);
        Assert.Contains("public bool CanPromoteRuntimeProof", builderConfigSnapshot);
        Assert.Contains("public bool CanDeleteDeferredRecord", builderConfigSnapshot);
        Assert.Contains("public sealed class TensorRtBuilderConfigSerializedPluginSnapshot", builderConfigSerializedPluginSnapshot);

        string publicText = string.Concat(
            engineApi,
            engineProfileApi,
            engineBindingReport,
            engineTensorBinding,
            engineDeploymentSnapshot,
            engineInspector,
            layerTensorMetadata,
            layerTensorMetadataModel,
            builderConfigApi,
            builderConfigSnapshot,
            builderConfigSerializedPluginSnapshot);

        Assert.DoesNotContain("public IntPtr", publicText);
        Assert.DoesNotContain("public nint", publicText);
        Assert.Contains("borrowed pointer public exposure", candidateText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("borrowed pointer public exposure implemented", candidateText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EngineAndBuilderEvidenceHasNativeSmokeAndQualityCoverage()
    {
        string nativeTrt8 = ReadSource("native", "src", "tensorrt", "v8", "api.cpp") +
            ReadSource("native", "src", "tensorrt", "v8", "modules", "builder", "builder_config.inc") +
            ReadSource("native", "src", "tensorrt", "v8", "modules", "parser", "parser_inspector.inc");
        string nativeTrt10 = ReadSource("native", "src", "tensorrt", "v10", "modules", "builder", "builder_config.inc") +
            ReadSource("native", "src", "tensorrt", "v10", "modules", "deployment", "engine_profile_tensor_values.inc") +
            ReadSource("native", "src", "tensorrt", "v10", "modules", "parser", "parser_inspector.inc");
        string nativeTrt11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "diagnostics.inc");
        string smokeTensorRt = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");
        string smokeNetworkBuilder = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string engineTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "EngineAndRnnReadonlyDiagnosticsTests.cs");
        string builderTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "BuilderConfigScalarControlsTests.cs");

        Assert.Contains("copy_engine_information", nativeTrt8);
        Assert.Contains("getLayerInformation", nativeTrt8 + nativeTrt10 + nativeTrt11);
        Assert.Contains("copy_string_to_buffer", nativeTrt11);
        Assert.Contains("engine_payload->getProfileTensorValues(", nativeTrt10);
        Assert.Contains("engine_payload->getProfileTensorValuesV2(", nativeTrt10);
        Assert.Contains("config_payload->getNbPluginsToSerialize()", nativeTrt8 + nativeTrt10 + nativeTrt11);
        Assert.Contains("config_payload->getFlags()", nativeTrt8 + nativeTrt10);

        Assert.Contains("engine.GetBindingReport", smokeTensorRt);
        Assert.Contains("engine.GetDeploymentSnapshot", smokeTensorRt);
        Assert.Contains("EngineDeploymentSummary=", smokeTensorRt);
        Assert.Contains("inspector.GetLayerInformation", smokeTensorRt);
        Assert.Contains("config.GetDeploymentSnapshot()", smokeNetworkBuilder + smokeTensorRt);
        Assert.Contains("BuilderConfigDeploymentSummary=", smokeTensorRt);
        Assert.Contains("config.SetPluginsToSerialize", smokeNetworkBuilder);
        Assert.Contains("config.GetSerializedPluginSnapshot", smokeNetworkBuilder);
        Assert.Contains("config.GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace)", smokeNetworkBuilder + smokeTensorRt);

        Assert.Contains("ManagedInteropAndPublicApiExposeCompatibilityQueriesWithoutRawPointers", engineTests);
        Assert.Contains("NetworkBuilderSmokeCoversScalarControlProbe", builderTests);
        Assert.Contains("InterfaceCoverageMatrixSeparatesRealImplementationFromDeferredHistory", builderTests);
    }

    private static JsonElement FindCandidate(JsonElement groups, string candidateId)
    {
        foreach (JsonProperty group in groups.EnumerateObject())
        {
            foreach (JsonElement candidate in group.Value.EnumerateArray())
            {
                if (candidate.GetProperty("candidateId").GetString() == candidateId)
                {
                    return candidate;
                }
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }

    private static void AssertCandidateImplemented(JsonElement candidate)
    {
        Assert.Equal("implemented-with-pointer-free-wrapper", candidate.GetProperty("implementationStatus").GetString());
        Assert.True(candidate.TryGetProperty("implementationEvidence", out JsonElement evidence));
        Assert.True(evidence.GetProperty("nativeSources").GetArrayLength() >= 3);
        Assert.True(evidence.GetProperty("managedSources").GetArrayLength() >= 4);
        Assert.True(evidence.GetProperty("smokeSources").GetArrayLength() >= 1);
        Assert.True(evidence.GetProperty("qualityTests").GetArrayLength() >= 1);
        Assert.True(evidence.GetProperty("publicSurface").GetArrayLength() >= 4);
        Assert.Contains("not exposed", evidence.GetProperty("ownershipBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertEvidenceContains(JsonElement candidate, string arrayName, string expected)
    {
        JsonElement evidence = candidate.GetProperty("implementationEvidence");
        foreach (JsonElement item in evidence.GetProperty(arrayName).EnumerateArray())
        {
            if (item.GetString() == expected)
            {
                return;
            }
        }

        throw new InvalidOperationException($"Expected {expected} in {candidate.GetProperty("candidateId").GetString()} evidence {arrayName}.");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
