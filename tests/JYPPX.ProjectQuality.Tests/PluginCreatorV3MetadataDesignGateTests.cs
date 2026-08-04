using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PluginCreatorV3MetadataDesignGateTests
{
    [Fact]
    public void PluginCreatorV3MetadataGateUsesCopiedInventoryAndDoesNotExposePointers()
    {
        TensorRtPluginCreatorV3MetadataDesignGateResult gate =
            TensorRtPluginCreatorV3MetadataDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("copied string, copied field metadata, and managed interface metadata snapshot", gate.RequiredOutputMode);
        Assert.Contains("IPluginCreatorV3One", gate.CandidateInterfaces);
        Assert.Contains("IVersionedInterface", gate.CandidateInterfaces);
        Assert.Contains("IPluginCreatorV3One::getPluginName", gate.CandidateMethods);
        Assert.Contains("IPluginCreatorV3One::getPluginVersion", gate.CandidateMethods);
        Assert.Contains("IPluginCreatorV3One::getPluginNamespace", gate.CandidateMethods);
        Assert.Contains("IPluginCreatorV3One::getFieldNames", gate.CandidateMethods);
        Assert.Contains("IPluginCreatorV3One::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IVersionedInterface::getInterfaceInfo", gate.CandidateMethods);
        Assert.Equal(6, gate.CandidateMethodCount);
        Assert.Contains("plugin registry inventory snapshots", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=6", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("RequiredOutputMode=copied string, copied field metadata, and managed interface metadata snapshot", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.DirectCreatePluginRowsDeferred);
        Assert.True(gate.DirectBorrowedCreatorListRowsDeferred);
        Assert.False(gate.PluginCreatorPointerExposed);
        Assert.False(gate.BorrowedPluginCreatorHandleExposed);
        Assert.False(gate.PluginInstanceCreationEnabled);
        Assert.False(gate.PluginResourceOwnershipControlEnabled);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void PluginCreatorV3MetadataCandidateListRecordsSafeAlternativeEvidence()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement pluginFieldMetadata = document.RootElement.GetProperty("groups").GetProperty("pluginFieldMetadata");

        JsonElement candidate = FindCandidate(pluginFieldMetadata, "plugin-creator-v3-metadata-design-003");

        Assert.Equal("implemented-safe-alternative-design-gate-not-runtime-proof", candidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("copied string, copied field metadata, and managed interface metadata snapshot", candidate.GetProperty("outputMode").GetString());
        AssertCandidateMethods(
            candidate,
            "IPluginCreatorV3One::getPluginName",
            "IPluginCreatorV3One::getPluginVersion",
            "IPluginCreatorV3One::getPluginNamespace",
            "IPluginCreatorV3One::getFieldNames",
            "IPluginCreatorV3One::getInterfaceInfo",
            "IVersionedInterface::getInterfaceInfo");

        AssertEvidenceContains(candidate, "nativeSources", "native/src/tensorrt/common/plugin_registry_inventory.inc");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginCreatorV3MetadataDesignGate.cs");
        AssertEvidenceContains(candidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/PluginCreatorV3MetadataDesignGateTests.cs");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtPluginCreatorV3MetadataDesignGate.EvaluateKnownSurface");

        string ownership = candidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("copied", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("native creator pointers are not exposed", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("createPlugin", ownership, StringComparison.Ordinal);
    }

    private static JsonElement FindCandidate(JsonElement candidates, string candidateId)
    {
        foreach (JsonElement candidate in candidates.EnumerateArray())
        {
            if (candidate.GetProperty("candidateId").GetString() == candidateId)
            {
                return candidate;
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }

    private static void AssertCandidateMethods(JsonElement candidate, params string[] expectedMethods)
    {
        string[] actual = candidate.GetProperty("candidateMethods")
            .EnumerateArray()
            .Select(static item => item.GetString() ?? string.Empty)
            .ToArray();

        foreach (string expected in expectedMethods)
        {
            Assert.Contains(expected, actual);
        }
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
}
