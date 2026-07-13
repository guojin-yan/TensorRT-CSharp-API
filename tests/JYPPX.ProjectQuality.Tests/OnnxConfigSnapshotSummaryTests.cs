using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxConfigSnapshotSummaryTests
{
    [Fact]
    public void OnnxConfigExposesPointerFreeSnapshotAndSummary()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxConfig.cs");

        Assert.Contains("public TensorRtOnnxConfigSnapshot ToSnapshot()", source);
        Assert.Contains("public sealed class TensorRtOnnxConfigSnapshot", source);
        Assert.Contains("public sealed class TensorRtOnnxConfigSummary", source);
        Assert.Contains("public TensorRtOnnxConfigSummary ToSummary()", source);
        Assert.Contains("public bool HasModelFileName => ModelFileNameLength > 0;", source);
        Assert.Contains("public bool HasTextFileName => TextFileNameLength > 0;", source);
        Assert.Contains("public bool HasFullTextFileName => FullTextFileNameLength > 0;", source);
        Assert.Contains("ModelFileName.Length", source);
        Assert.Contains("TextFileName.Length", source);
        Assert.Contains("FullTextFileName.Length", source);
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public nint", source);
    }

    [Fact]
    public void OnnxConfigSnapshotUsesCopiedStringBridge()
    {
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxConfig.cs");
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxConfig.cs");

        Assert.Contains("ReadOnnxConfigString", interop);
        Assert.Contains("byte[] buffer = new byte[checked((int)required)];", interop);
        Assert.Contains("Encoding.UTF8.GetString(buffer, 0, length)", interop);
        Assert.Contains("ModelFileName,", source);
        Assert.Contains("TextFileName,", source);
        Assert.Contains("FullTextFileName,", source);
    }

    [Fact]
    public void OnnxToEngineSmokePrintsOnnxConfigEvidenceMarkers()
    {
        string program = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("using TensorRtOnnxConfig onnxConfig = new TensorRtOnnxConfig(line);", program);
        Assert.Contains("TensorRtOnnxConfigSnapshot onnxConfigSnapshot = onnxConfig.ToSnapshot();", program);
        Assert.Contains("TensorRtOnnxConfigSummary onnxConfigSummary = onnxConfigSnapshot.ToSummary();", program);
        Assert.Contains("OnnxConfigSnapshot={onnxConfigSnapshot}", program);
        Assert.Contains("OnnxConfigSummary={onnxConfigSummary}", program);
    }

    [Fact]
    public void OnnxConfigSafeScalarManifestsRemainCrossVersionTracked()
    {
        string trt10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-onnx-config-safe-scalar-controls.manifest.json");
        string trt11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-onnx-config-safe-scalar-controls.manifest.json");

        Assert.Contains("jyppx_trt10_onnx_config_create", trt10);
        Assert.Contains("jyppx_trt10_onnx_config_get_model_file_name", trt10);
        Assert.Contains("jyppx_trt10_onnx_config_get_full_text_file_name", trt10);
        Assert.Contains("jyppx_trt11_onnx_config_create", trt11);
        Assert.Contains("jyppx_trt11_onnx_config_get_model_file_name", trt11);
        Assert.Contains("jyppx_trt11_onnx_config_get_full_text_file_name", trt11);
    }

    [Fact]
    public void OnnxConfigSummaryDoesNotPromoteUnsafeOwnerApis()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxConfig.cs");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.DoesNotContain("IOnnxConfig*", source);
        Assert.DoesNotContain("borrowed pointer", source, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("IOnnxConfig", comparison);
        Assert.Contains("deferred", comparison, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
