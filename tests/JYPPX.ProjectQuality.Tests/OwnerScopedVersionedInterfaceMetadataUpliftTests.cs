using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerScopedVersionedInterfaceMetadataUpliftTests
{
    [Fact]
    public void ManifestsDeclareTwentyTwoPointerFreeEntriesAcrossTensorRt10And11()
    {
        foreach (string line in new[] { "10", "11" })
        {
            string manifest = ReadSource(
                "native",
                "manifests",
                "tensorrt",
                $"v{line}",
                $"trt{line}-owner-scoped-versioned-interface-metadata.manifest.json");
            using JsonDocument document = JsonDocument.Parse(manifest);
            JsonElement.ArrayEnumerator apis = document.RootElement.GetProperty("apis").EnumerateArray();
            JsonElement[] entries = apis.ToArray();

            Assert.Equal(11, entries.Length);
            Assert.All(entries, entry =>
            {
                Assert.Equal($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", entry.GetProperty("versionGuard").GetString());
                Assert.Equal("caller-owned", entry.GetProperty("ownership").GetString());
            });
            Assert.Contains($"trt{line}-runtime-get-error-recorder-versioned-metadata", manifest, StringComparison.Ordinal);
            Assert.Contains($"trt{line}-builder-config-get-progress-monitor-versioned-metadata", manifest, StringComparison.Ordinal);
            Assert.Contains($"trt{line}-execution-context-output-allocator-get-api-language", manifest, StringComparison.Ordinal);
            Assert.DoesNotContain("void*", manifest, StringComparison.Ordinal);
            Assert.DoesNotContain("\"managedType\": \"IntPtr\"", manifest, StringComparison.Ordinal);
            Assert.DoesNotContain("\"moduleManagedType\": \"IntPtr\"", manifest, StringComparison.Ordinal);
            Assert.DoesNotContain("nint", manifest, StringComparison.Ordinal);
            Assert.DoesNotContain("SafeHandle", manifest, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void NativeImplementationCopiesWithinOwnerScopeAndGuardsVendorCalls()
    {
        string native = ReadSource(
            "native",
            "src",
            "tensorrt",
            "common",
            "owner_scoped_versioned_interface_metadata.inc");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");

        Assert.Contains("copy_interface_info_to_buffer", native, StringComparison.Ordinal);
        Assert.Contains("object->getInterfaceInfo()", native, StringComparison.Ordinal);
        Assert.Contains("object->getAPILanguage()", native, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", native, StringComparison.Ordinal);
        Assert.Contains("report_vendor_exception", native, StringComparison.Ordinal);
        Assert.Contains("validate_handle", native, StringComparison.Ordinal);
        Assert.Contains("get_payload<TOwner>", native, StringComparison.Ordinal);
        Assert.DoesNotContain("TInterface**", native, StringComparison.Ordinal);
        Assert.DoesNotContain("out_interface", native, StringComparison.Ordinal);
        Assert.Contains("owner_scoped_versioned_interface_metadata.inc", trt10Api, StringComparison.Ordinal);
        Assert.Contains("owner_scoped_versioned_interface_metadata.inc", trt11Api, StringComparison.Ordinal);
        Assert.DoesNotContain("owner_scoped_versioned_interface_metadata.inc", ReadSource("native", "src", "tensorrt", "v8", "api.cpp"), StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsTypedPointerFreeAndPreservesVersionGuards()
    {
        string model = ReadSource("src", "JYPPX.TensorRtSharp", "Interfaces", "TensorRtVersionedInterfaceMetadata.cs");
        string surface = ReadSource("src", "JYPPX.TensorRtSharp", "Interfaces", "TensorRtOwnerScopedVersionedInterfaceMetadata.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OwnerScopedVersionedInterfaceMetadata.cs");

        Assert.Contains("public sealed class TensorRtVersionedInterfaceMetadata", model, StringComparison.Ordinal);
        Assert.Contains("public TensorRtInterfaceInfo InterfaceInfo", model, StringComparison.Ordinal);
        Assert.Contains("public TensorRtApiLanguage ApiLanguage", model, StringComparison.Ordinal);
        Assert.Contains("public bool PointerFreeCopiedMetadata => true", model, StringComparison.Ordinal);
        Assert.Contains("public bool RetainsNativeInterface => false", model, StringComparison.Ordinal);
        Assert.Contains("TryGetErrorRecorderVersionedMetadata", surface, StringComparison.Ordinal);
        Assert.Contains("TryGetProgressMonitorVersionedMetadata", surface, StringComparison.Ordinal);
        Assert.Contains("TryGetOutputAllocatorVersionedMetadata", surface, StringComparison.Ordinal);
        Assert.Contains("TryGetTemporaryStorageAllocatorVersionedMetadata", surface, StringComparison.Ordinal);
        Assert.Contains("TryGetDebugListenerVersionedMetadata", surface, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt10", interop, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt11", interop, StringComparison.Ordinal);
        Assert.Contains("UnsupportedOwnerScopedVersionedMetadataLine", interop, StringComparison.Ordinal);

        string publicSurface = model + surface;
        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratedBindingsContainAllTwentyTwoEntrypoints()
    {
        string generated = ReadSource(
            "src",
            "JYPPX.TensorRtSharp",
            "Internal",
            "Interop",
            "Generated",
            "NativeMethodsTensorRt.Generated.g.cs");

        foreach (string line in new[] { "10", "11" })
        {
            foreach (string suffix in ExpectedEntryPointSuffixes)
            {
                Assert.Contains($"jyppx_trt{line}_{suffix}", generated, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void CoverageAliasesPreferSafeEntriesAndPreserveDeferredHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("\"IPluginV2Ext::getTensorRTVersion\" = @(\"id:*plugin-v2-layer-get-tensor-rt-version\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IVersionedInterface::getAPILanguage\" = @(\"id:*versioned-metadata\", \"id:*get-api-language\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IVersionedInterface::getInterfaceInfo\" = @(\"id:*versioned-metadata\", \"id:*get-interface-info\")", script, StringComparison.Ordinal);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int priorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", priorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && priorityStart > matcherStart && heuristicStart > priorityStart);
        string priorityBlock = script.Substring(priorityStart, heuristicStart - priorityStart);
        Assert.Contains("IVersionedInterface::getAPILanguage", priorityBlock, StringComparison.Ordinal);
        Assert.Contains("IVersionedInterface::getInterfaceInfo", priorityBlock, StringComparison.Ordinal);

        foreach (string expected in new[]
                 {
                     "\"IPluginV2Ext\",\"getTensorRTVersion\",\"IPluginV2Ext::getTensorRTVersion\",\"plugin\",\"implemented-with-deferred-history\"",
                     "\"IPluginV2IOExt\",\"getTensorRTVersion\",\"IPluginV2IOExt::getTensorRTVersion\",\"plugin\",\"implemented-with-deferred-history\"",
                     "\"IVersionedInterface\",\"getAPILanguage\",\"IVersionedInterface::getAPILanguage\",\"other\",\"implemented-with-deferred-history\"",
                     "\"IVersionedInterface\",\"getInterfaceInfo\",\"IVersionedInterface::getInterfaceInfo\",\"other\",\"implemented-with-deferred-history\""
                 })
        {
            Assert.Contains(expected, comparison, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SmokeAndPackageConsumerExerciseOwnerScopedMetadata()
    {
        string callbackSmoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string progressSmoke = ReadSource("smoke", "ManagedProgressMonitorSmokeRunner", "Program.cs");
        string packageConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("TryGetErrorRecorderVersionedMetadata", callbackSmoke, StringComparison.Ordinal);
        Assert.Contains("TryGetOutputAllocatorVersionedMetadata", callbackSmoke, StringComparison.Ordinal);
        Assert.Contains("TryGetTemporaryStorageAllocatorVersionedMetadata", callbackSmoke, StringComparison.Ordinal);
        Assert.Contains("TryGetDebugListenerVersionedMetadata", callbackSmoke, StringComparison.Ordinal);
        Assert.Contains("TryGetProgressMonitorVersionedMetadata", progressSmoke, StringComparison.Ordinal);
        Assert.Contains("ManagedProgressMonitorOwnerScopedMetadata", progressSmoke, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtVersionedInterfaceMetadata)", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtExecutionContext.TryGetDebugListenerVersionedMetadata)", packageConsumer, StringComparison.Ordinal);
    }

    private static readonly string[] ExpectedEntryPointSuffixes =
    {
        "runtime_get_error_recorder_versioned_metadata",
        "refitter_get_error_recorder_versioned_metadata",
        "engine_get_error_recorder_versioned_metadata",
        "execution_context_get_error_recorder_versioned_metadata",
        "builder_get_error_recorder_versioned_metadata",
        "network_get_error_recorder_versioned_metadata",
        "engine_inspector_get_error_recorder_versioned_metadata",
        "execution_context_get_output_allocator_api_language",
        "execution_context_get_temporary_storage_allocator_api_language",
        "execution_context_get_debug_listener_api_language",
        "builder_config_get_progress_monitor_versioned_metadata"
    };

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }
}
