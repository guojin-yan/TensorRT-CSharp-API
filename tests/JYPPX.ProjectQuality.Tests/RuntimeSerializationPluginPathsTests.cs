using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeSerializationPluginPathsTests
{
    [Fact]
    public void TensorRt10PluginSerializationPathsAreLiftedWithoutDeletingDeferredRecords()
    {
        string manifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-serialization-plugin-paths.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string source = ReadSource("native", "src", "tensorrt", "v10", "modules", "builder", "builder_config.inc");

        Assert.Contains("trt10-builder-config-clear-plugins-to-serialize", manifest);
        Assert.Contains("trt10-builder-config-get-plugin-to-serialize-count", manifest);
        Assert.Contains("trt10-builder-config-get-nb-plugins-to-serialize", manifest);
        Assert.Contains("trt10-builder-config-get-plugin-to-serialize", manifest);
        Assert.Contains("trt10-builder-config-set-plugins-to-serialize", manifest);
        Assert.Contains("jyppx_trt10_builder_config_get_plugin_to_serialize_count", manifest);
        Assert.Contains("jyppx_trt10_builder_config_get_nb_plugins_to_serialize", manifest);
        Assert.Contains("jyppx_trt10_builder_config_set_plugins_to_serialize", manifest);
        Assert.Contains("\"type\": \"char*\", \"direction\": \"out\", \"managedType\": \"byte[]\"", manifest);
        Assert.Contains("\"type\": \"const char**\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);

        Assert.Contains("trt10-builder-config-get-nb-plugins-to-serialize-deferred", deferredManifest);
        Assert.Contains("trt10-builder-config-get-plugin-to-serialize-deferred", deferredManifest);
        Assert.Contains("trt10-builder-config-set-plugins-to-serialize-deferred", deferredManifest);

        Assert.Contains("config_payload->setPluginsToSerialize(nullptr, 0);", source);
        Assert.Contains("*out_count = config_payload->getNbPluginsToSerialize();", source);
        Assert.Contains("jyppx_trt10_builder_config_get_nb_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)", source);
        Assert.Contains("return jyppx_trt10_builder_config_get_plugin_to_serialize_count(config, out_count);", source);
        Assert.Contains("copy_string_to_buffer(config_payload->getPluginToSerialize(index)", source);
        Assert.Contains("config_payload->setPluginsToSerialize(paths, path_count);", source);
        Assert.Contains("validate_c_string(paths[i], \"plugin_path\")", source);
    }

    [Fact]
    public void ManagedPluginSerializationPathsAreTensorRt10AndTensorRt11Only()
    {
        string diagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11Diagnostics.cs");
        string fourteenthBatch = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11FourteenthBatch.cs");
        string publicDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilderConfig.Trt11Diagnostics.cs");
        string publicSerialization = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilderConfig.Trt11PluginSerialization.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtBuilderConfigSerializedPluginSnapshot.cs");

        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_clear_plugins_to_serialize", diagnostics);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_nb_plugins_to_serialize", diagnostics);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_nb_plugins_to_serialize", diagnostics);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_plugin_to_serialize", diagnostics);
        Assert.Contains("TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported", diagnostics);
        Assert.Contains("EnsureTensorRt10Or11(line, nameof(GetBuilderConfigPluginToSerialize))", diagnostics);

        Assert.Contains("EnsureTensorRt10Or11(line, nameof(SetBuilderConfigPluginsToSerialize))", fourteenthBatch);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_plugins_to_serialize", fourteenthBatch);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_plugins_to_serialize", fourteenthBatch);

        Assert.Contains("TensorRT 10/11", publicDiagnostics);
        Assert.Contains("TensorRT 10/11", publicSerialization);
        Assert.Contains("public sealed class TensorRtBuilderConfigSerializedPluginSnapshot", snapshot);
        Assert.Contains("public TensorRtBuilderConfigSerializedPluginSnapshot GetSerializedPluginSnapshot()", publicDiagnostics);
        Assert.Contains("public bool TryGetSerializedPluginSnapshot(out TensorRtBuilderConfigSerializedPluginSnapshot snapshot, out string diagnostic)", publicDiagnostics);
        Assert.Contains("SerializedPluginPathCountCompatibility", publicDiagnostics);
        Assert.Contains("TryGetPluginsToSerialize", publicDiagnostics);
        Assert.Contains("public IReadOnlyList<string> PluginLibraryPaths", snapshot);
        Assert.Contains("public bool HasPathInventory", snapshot);
        Assert.Contains("public string Diagnostic", snapshot);
        Assert.Contains("does not load plugin libraries, create plugins, deserialize plugins, or expose TensorRT-owned pointers", publicDiagnostics);
        Assert.DoesNotContain("public IntPtr", publicDiagnostics + publicSerialization + snapshot);
        Assert.DoesNotContain("public nint", publicDiagnostics + publicSerialization + snapshot);
    }

    [Fact]
    public void PluginSerializationPathApisAreDeclaredInPublicNativeHeaders()
    {
        AssertPluginSerializationHeaderDeclarations("10");
        AssertPluginSerializationHeaderDeclarations("11");
    }

    [Fact]
    public void PluginSerializationPathsSmokeCoversSetGetClearWithoutBuildingEngine()
    {
        string project = ReadSource("smoke", "PluginSerializationPathsSmokeRunner", "PluginSerializationPathsSmokeRunner.csproj");
        string program = ReadSource("smoke", "PluginSerializationPathsSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("PluginSerializationPathsSmokeRunner", smokeReadme);
        Assert.Contains("PluginSerializationPathsSmokeRunner.csproj", solution);
        Assert.Contains("<ProjectReference Include=\"..\\..\\src\\JYPPX.TensorRtSharp\\JYPPX.TensorRtSharp.csproj\" />", project);
        Assert.Contains("--dependency-probe-only", program);
        Assert.Contains("SetPluginsToSerialize", program);
        Assert.Contains("GetPluginsToSerialize", program);
        Assert.Contains("GetSerializedPluginSnapshot", program);
        Assert.Contains("TryGetSerializedPluginSnapshot", program);
        Assert.Contains("ClearPluginsToSerialize", program);
        Assert.Contains("Snapshot=", program);
        Assert.Contains("ClearedSnapshot=", program);
        Assert.Contains("Skipped=True Reason=DependencyProbeOnly", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertPluginSerializationHeaderDeclarations(string line)
    {
        string header = ReadSource("native", "include", "jyppx", "tensorrt", $"trt{line}.h");

        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_builder_config_clear_plugins_to_serialize", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_builder_config_set_plugins_to_serialize", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_builder_config_get_plugin_to_serialize_count", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_builder_config_get_nb_plugins_to_serialize", header);
        Assert.Contains($"JYPPX_C_API(JYPPX_StatusCode) jyppx_trt{line}_builder_config_get_plugin_to_serialize", header);
    }
}
