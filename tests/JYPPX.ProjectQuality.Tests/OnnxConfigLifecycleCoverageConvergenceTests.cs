using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxConfigLifecycleCoverageConvergenceTests
{
    [Fact]
    public void CoverageAliasesPreferRealCreateAndDestroyEntriesBeforeHeuristics()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"Global::createONNXConfig\" = @(\"id:*onnx-config-create\")", script);
        Assert.Contains("\"IOnnxConfig::destroy\" = @(\"id:*trt-object-destroy\")", script);
        Assert.Contains("\"Global::createONNXConfig\" = @(\"id:*onnx-config-create-deferred\", \"id:*global-create-onnx-config-deferred\")", script);
        Assert.Contains("\"IOnnxConfig::destroy\" = @(\"id:*onnx-config-destroy-deferred\")", script);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int priorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", priorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && priorityStart > matcherStart && heuristicStart > priorityStart);

        string priorityBlock = script.Substring(priorityStart, heuristicStart - priorityStart);
        Assert.Contains("Global::createONNXConfig", priorityBlock);
        Assert.Contains("IOnnxConfig::destroy", priorityBlock);
    }

    [Fact]
    public void CoverageReportsThreeFactoriesAndTrt8DestroyWithDeferredHistory()
    {
        string comparison = ReadSource(
            "artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Equal(6, CountRows(
            comparison,
            "\"Global\",\"createONNXConfig\",\"Global::createONNXConfig\",\"global\",\"implemented-with-deferred-history\""));
        Assert.Contains(
            "\"IOnnxConfig\",\"destroy\",\"IOnnxConfig::destroy\",\"onnx-parser\",\"implemented-with-deferred-history\"",
            comparison);
    }

    [Fact]
    public void NativeCreationTransfersOwnershipOnlyAfterHandleCreationSucceeds()
    {
        string cmake = ReadSource("CMakeLists.txt");
        string controls = ReadSource(
            "native", "src", "tensorrt", "common", "onnx_config_controls.inc");

        int ownerStart = controls.IndexOf(
            "std::unique_ptr<BridgeOwnedOnnxConfig> config_owner", StringComparison.Ordinal);
        int handleCreate = controls.IndexOf("create_handle_with_payload(", ownerStart, StringComparison.Ordinal);
        int release = controls.IndexOf("config_owner.release();", handleCreate, StringComparison.Ordinal);
        Assert.True(ownerStart >= 0 && handleCreate > ownerStart && release > handleCreate);
        Assert.Contains("config_owner.get()", controls);
        Assert.DoesNotContain("config.release()", controls);
        Assert.Contains("JYPPX_HAS_TENSORRT_ONNX_CONFIG_VALUE", cmake);
        Assert.Contains("EXISTS \"${TensorRT_ROOT}/include/NvOnnxConfig.h\"", cmake);
        Assert.Contains("JYPPX_HAS_TENSORRT_ONNX_CONFIG=${JYPPX_HAS_TENSORRT_ONNX_CONFIG_VALUE}", cmake);
        Assert.DoesNotContain(
            "JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER",
            controls);
        Assert.Contains(
            "JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNX_CONFIG",
            controls);

        foreach (string line in new[] { "v8", "v10", "v11" })
        {
            string api = ReadSource("native", "src", "tensorrt", line, "api.cpp");
            Assert.Contains("#if JYPPX_HAS_TENSORRT_ONNX_CONFIG", api);
            Assert.Contains("#include <NvOnnxConfig.h>", api);
            Assert.Contains("#ifndef JYPPX_HAS_TENSORRT_ONNX_CONFIG", api);
        }
    }

    [Fact]
    public void RealAndDeferredManifestsCoexistAcrossIndependentVersionGuards()
    {
        foreach (string line in new[] { "8", "10", "11" })
        {
            string real = ReadSource(
                "native", "manifests", "tensorrt", $"v{line}",
                $"trt{line}-onnx-config-safe-scalar-controls.manifest.json");
            Assert.Contains($"trt{line}-onnx-config-create", real);
            Assert.Contains($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", real);
        }

        Assert.Contains(
            "trt8-onnx-config-destroy-deferred",
            ReadSource(
                "native", "manifests", "tensorrt", "v8",
                "trt8-cross-version-eighth-batch-onnx-config-parser-deferred.manifest.json"));
        Assert.Contains(
            "trt10-onnx-config-create-deferred",
            ReadSource(
                "native", "manifests", "tensorrt", "v10",
                "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json"));
        Assert.Contains(
            "trt11-onnx-config-create-deferred",
            ReadSource(
                "native", "manifests", "tensorrt", "v11",
                "trt11-forty-fourth-batch-global-deferred.manifest.json"));
    }

    [Fact]
    public void ManagedLifecycleAndPackageConsumerRemainPointerFree()
    {
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxConfig.cs");
        string safeHandle = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Handles", "SafeTensorRtObjectHandle.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("public sealed class TensorRtOnnxConfig : IDisposable", wrapper);
        Assert.Contains("_handle.Dispose();", wrapper);
        Assert.Contains("jyppx_trt_object_destroy(handle)", safeHandle);
        Assert.Contains("static line => new TensorRtOnnxConfig(line)", consumer);
        Assert.Contains("static config => config.ToSnapshot()", consumer);
        Assert.Contains("static snapshot => snapshot.ToSummary()", consumer);
        Assert.Contains("static config => config.Dispose()", consumer);
        Assert.Contains("onnx-config-owned-lifecycle", consumer);
        Assert.DoesNotContain("public IntPtr", wrapper);
        Assert.DoesNotContain("public nint", wrapper);
        Assert.DoesNotContain("public SafeHandle", wrapper);
    }

    private static int CountRows(string text, string rowPrefix)
    {
        int count = 0;
        int offset = 0;
        while ((offset = text.IndexOf(rowPrefix, offset, StringComparison.Ordinal)) >= 0)
        {
            ++count;
            offset += rowPrefix.Length;
        }
        return count;
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
