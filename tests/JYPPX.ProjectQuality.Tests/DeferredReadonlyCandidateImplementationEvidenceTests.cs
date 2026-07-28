using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredReadonlyCandidateImplementationEvidenceTests
{
    [Fact]
    public void PluginReadonlyCandidatesAreLinkedToImplementedPointerFreeInventory()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        string candidateText = File.ReadAllText(candidatePath);
        using JsonDocument document = JsonDocument.Parse(candidateText);

        JsonElement groups = document.RootElement.GetProperty("groups");
        JsonElement fieldCandidate = FindCandidate(groups, "plugin-field-metadata-001");
        JsonElement identityCandidate = FindCandidate(groups, "plugin-creator-identity-002");

        AssertCandidateImplemented(fieldCandidate);
        AssertCandidateImplemented(identityCandidate);

        AssertEvidenceContains(fieldCandidate, "publicSurface", "TensorRtPluginRegistryInventory.GetFieldSummaries");
        AssertEvidenceContains(fieldCandidate, "publicSurface", "TensorRtPluginRegistryInventory.GetDiagnostics");
        AssertEvidenceContains(identityCandidate, "publicSurface", "TensorRtPluginRegistryInventory.GetCreatorSummaries");
        AssertEvidenceContains(identityCandidate, "publicSurface", "TensorRtPluginRegistryInventory.TryFindCreator");

        string inventoryModels = ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginRegistryInventory.cs");
        string builderInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilder.PluginRegistryInventory.cs");
        string runtimeInventory = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.PluginRegistryInventory.cs");
        string environmentProbe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.cs");
        string builderInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.PluginRegistryInventory.cs");
        string runtimeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.RuntimePluginRegistryInventory.cs");
        string globalInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.GlobalRuntimePluginProbe.cs");
        string capabilityInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.BuilderCapabilityPluginRegistry.cs");
        string smoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string nativeCommon = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string nativeGlobal = ReadSource("native", "src", "tensorrt", "common", "global_runtime_plugin_probe.inc");
        string nativeTrt8 = ReadSource("native", "src", "tensorrt", "v8", "modules", "plugin", "trt8_plugin_registry_inventory.inc");

        Assert.Contains("public IReadOnlyList<TensorRtPluginCreatorSummary> GetCreatorSummaries", inventoryModels);
        Assert.Contains("public IReadOnlyList<TensorRtPluginFieldSummary> GetFieldSummaries", inventoryModels);
        Assert.Contains("public TensorRtPluginRegistryInventoryDiagnostics GetDiagnostics", inventoryModels);
        Assert.Contains("public bool TryFindCreator", inventoryModels);
        Assert.Contains("Pointer-free", inventoryModels, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("public IntPtr", inventoryModels + builderInventory + runtimeInventory + environmentProbe);
        Assert.DoesNotContain("public nint", inventoryModels + builderInventory + runtimeInventory + environmentProbe);

        foreach (string interop in new[] { builderInterop, runtimeInterop, globalInterop, capabilityInterop })
        {
            Assert.Contains("Get", interop);
            Assert.Contains("PluginCreator", interop);
            Assert.Contains("FieldMetadata", interop);
            Assert.Contains("new TensorRtPluginCreatorInfo(", interop);
            Assert.Contains("fields.Add(new TensorRtPluginFieldInfo", interop);
        }

        Assert.Contains("TryGetPluginCreator", smoke);
        Assert.Contains("GetCreatorSummaries", smoke);
        Assert.Contains("GetFieldSummaries", smoke);
        Assert.Contains("GetDiagnostics", smoke);

        Assert.Contains("copy_string_to_buffer", nativeCommon + nativeGlobal + nativeTrt8);
        Assert.Contains("getFieldNames", nativeCommon + nativeGlobal + nativeTrt8);
        Assert.Contains("getPluginName", nativeCommon + nativeGlobal + nativeTrt8);
        Assert.Contains("getPluginVersion", nativeCommon + nativeGlobal + nativeTrt8);
        Assert.Contains("getPluginNamespace", nativeCommon + nativeGlobal + nativeTrt8);

        Assert.Contains("plugin instance create", candidateText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("plugin enqueue implemented", candidateText, StringComparison.OrdinalIgnoreCase);
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
        Assert.True(evidence.GetProperty("nativeSources").GetArrayLength() >= 2);
        Assert.True(evidence.GetProperty("managedSources").GetArrayLength() >= 4);
        Assert.True(evidence.GetProperty("smokeSources").GetArrayLength() >= 1);
        Assert.True(evidence.GetProperty("qualityTests").GetArrayLength() >= 1);
        Assert.Contains("native pointers", evidence.GetProperty("ownershipBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
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
