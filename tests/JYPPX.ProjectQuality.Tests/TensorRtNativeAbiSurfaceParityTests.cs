using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtNativeAbiSurfaceParityTests
{
    [Theory]
    [InlineData("8", 900)]
    [InlineData("10", 1000)]
    [InlineData("11", 1100)]
    public void EveryManifestEntryPointHasAnExportedHeaderDeclaration(string line, int minimumExpectedCount)
    {
        string manifestRoot = Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt", "v" + line);
        string header = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "native", "include", "jyppx", "tensorrt", "trt" + line + ".h"));
        HashSet<string> entryPoints = new HashSet<string>(StringComparer.Ordinal);

        foreach (string manifestPath in Directory.GetFiles(manifestRoot, "*.manifest.json").OrderBy(static path => path, StringComparer.Ordinal))
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
            foreach (JsonElement api in document.RootElement.GetProperty("apis").EnumerateArray())
            {
                string entryPoint = api.GetProperty("entryPoint").GetString()!;
                Assert.False(string.IsNullOrWhiteSpace(entryPoint));
                entryPoints.Add(entryPoint);
            }
        }

        Assert.True(entryPoints.Count >= minimumExpectedCount, $"TRT{line} manifest inventory unexpectedly shrank to {entryPoints.Count} entries.");
        foreach (string entryPoint in entryPoints)
        {
            string escapedEntryPoint = Regex.Escape(entryPoint);
            string explicitPattern = @"JYPPX_C_API\s*\(\s*JYPPX_StatusCode\s*\)\s+" + escapedEntryPoint + @"\s*\(";
            string macroPattern = @"(?m)^\s*[A-Z][A-Z0-9_]*_DECL\s*\(\s*" + escapedEntryPoint + @"\s*\)\s*;?\s*$";
            bool declared =
                Regex.IsMatch(header, explicitPattern, RegexOptions.CultureInvariant) ||
                Regex.IsMatch(header, macroPattern, RegexOptions.CultureInvariant);
            Assert.True(declared, $"TRT{line} manifest entry point '{entryPoint}' has no exported header declaration.");
        }
    }

    [Fact]
    public void Trt10BuilderRegistryExistenceRouteIsDeclaredImplementedAndConsumed()
    {
        string header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string source = ReadSource("native", "src", "tensorrt", "common", "plugin_registry_inventory.inc");
        string api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.PluginRegistryInventory.cs");

        Assert.Contains("JYPPX_C_API(JYPPX_StatusCode) jyppx_trt10_builder_plugin_registry_exists", header, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_exists)", source, StringComparison.Ordinal);
        Assert.Contains("#define JYPPX_TRT_PLUGIN_PREFIX jyppx_trt10_", api, StringComparison.Ordinal);
        Assert.Contains("NativeMethodsTensorRt.jyppx_trt10_builder_plugin_registry_exists", interop, StringComparison.Ordinal);
    }

    [Fact]
    public void Trt11RefitterErrorRecorderSnapshotIsCopiedAndExceptionContained()
    {
        string source = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "runtime_serialization_refit.inc");

        Assert.Contains("jyppx_trt11_refitter_get_error_recorder_snapshot_info", source, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_refitter_get_error_recorder_error", source, StringComparison.Ordinal);
        Assert.Contains("trt11_refitter_get_error_recorder_snapshot_info_with_seh_guard", source, StringComparison.Ordinal);
        Assert.Contains("trt11_refitter_get_error_recorder_error_with_seh_guard", source, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", source, StringComparison.Ordinal);
        Assert.Contains("trt11_runtime_control_exception", source, StringComparison.Ordinal);
        Assert.Contains("trt11_runtime_control_unknown_exception", source, StringComparison.Ordinal);
        Assert.Contains("copy_c_string(recorder->getErrorDesc(index)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("out_recorder", source, StringComparison.Ordinal);
    }

    [Fact]
    public void ScriptAndCiEnforceStaticAndOptionalPeExportParityWithoutPublishing()
    {
        string script = ReadSource("eng", "Test-TensorRtNativeAbiSurface.ps1");
        string workflow = ReadSource(".github", "workflows", "release-quality-gate.yml");

        Assert.Contains("Get-ManifestEntryPoints", script, StringComparison.Ordinal);
        Assert.Contains("Resolve-DumpbinPath", script, StringComparison.Ordinal);
        Assert.Contains("/exports", script, StringComparison.Ordinal);
        Assert.Contains("missingDeclarationCount", script, StringComparison.Ordinal);
        Assert.Contains("missingExportCount", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("Test-TensorRtNativeAbiSurface.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("artifacts/native-abi/**", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("release upload", script, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }
}
