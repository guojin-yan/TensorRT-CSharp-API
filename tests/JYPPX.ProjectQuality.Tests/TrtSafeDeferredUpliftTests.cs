using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtSafeDeferredUpliftTests
{
    [Fact]
    public void SelectedVendorEntriesKeepDeferredHistoryAndVersionGuards()
    {
        string[] manifests =
        {
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-safe-deferred-uplift.manifest.json"),
            ReadSource("native", "manifests", "tensorrt", "v10", "trt10-safe-deferred-uplift.manifest.json"),
            ReadSource("native", "manifests", "tensorrt", "v11", "trt11-safe-deferred-uplift.manifest.json")
        };

        string[] deferred =
        {
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json"),
            ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-eighth-batch-onnx-config-parser-deferred.manifest.json"),
            ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json")
        };

        Assert.Contains("trt8-global-init-lib-nvinfer-plugins", manifests[0]);
        Assert.Contains("trt8-onnx-parser-parse-with-weight-descriptors", manifests[0]);
        Assert.Contains("trt10-global-init-lib-nvinfer-plugins", manifests[1]);
        Assert.Contains("trt10-onnx-parser-parse-with-weight-descriptors", manifests[1]);
        Assert.Contains("trt11-global-init-lib-nvinfer-plugins", manifests[2]);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8", manifests[0]);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10", manifests[1]);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", manifests[2]);
        Assert.Contains("trt8-global-init-lib-nvinfer-plugins-deferred", deferred[0]);
        Assert.Contains("trt8-parser-parse-with-weight-descriptors-deferred", deferred[1]);
        Assert.Contains("trt10-global-init-lib-nvinfer-plugins-deferred", deferred[2]);
        Assert.Contains("trt10-parser-parse-with-weight-descriptors-deferred", deferred[2]);
    }

    [Fact]
    public void NativeAndManagedBoundariesContainTheSafeUpliftContract()
    {
        string pluginNative = ReadSource("native", "src", "tensorrt", "common", "safe_deferred_plugin_initialization.inc");
        string parserNative = ReadSource("native", "src", "tensorrt", "common", "safe_deferred_onnx_parse.inc");
        string bridge =
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Plugins", "NativeBridgeApi.PluginInitialization.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.OnnxWeightDescriptorParsing.cs");
        string probe = ReadSource("src", "JYPPX.TensorRtSharp", "Diagnostics", "TensorRtEnvironmentProbe.PluginInitialization.cs");
        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelParsing.cs");
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string cmake = ReadSource("CMakeLists.txt");

        Assert.Contains("initLibNvInferPlugins", pluginNative);
        Assert.Contains("parseWithWeightDescriptors", parserNative);
        Assert.Contains("capture_vendor_seh_exception_code", pluginNative);
        Assert.Contains("capture_vendor_seh_exception_code", parserNative);
        Assert.Contains("catch (const std::exception& exception)", pluginNative);
        Assert.Contains("catch (const std::exception& exception)", parserNative);
        Assert.Contains("GCHandle.Alloc(modelData, GCHandleType.Pinned)", bridge);
        Assert.Contains("InitializeBuiltInPlugins", probe);
        Assert.Contains("ParseWithWeightDescriptors", parser);
        Assert.DoesNotContain("public IntPtr", probe + parser);
        Assert.DoesNotContain("public nint", probe + parser);
        Assert.Contains("IParser::parseWithWeightDescriptors", coverage);
        Assert.Contains("global-init-lib-nvinfer-plugins", coverage);
        Assert.Contains("TensorRT::nvinfer_plugin", cmake);
        Assert.Contains("nvinfer_plugin_10.dll", cmake);
        Assert.Contains("nvinfer_plugin_11.dll", cmake);
    }

    [Fact]
    public void SmokeRunnersExerciseInitializationAndLegacyParserPath()
    {
        string pluginSmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("InitializeBuiltInPlugins", pluginSmoke);
        Assert.Contains("BuiltInPluginInitialization Initialized=", pluginSmoke);
        Assert.Contains("ParseWithWeightDescriptors", onnxSmoke);
        Assert.Contains("LegacyWeightDescriptorParse Attempted=", onnxSmoke);
        Assert.Contains("Reason=RemovedByVendor", onnxSmoke);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
