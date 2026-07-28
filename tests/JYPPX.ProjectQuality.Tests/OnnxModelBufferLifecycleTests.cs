using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxModelBufferLifecycleTests
{
    [Fact]
    public void OnnxModelBufferManifestsPromoteSafeCallerOwnedInputs()
    {
        string trt10 = ReadTensorRtManifest("v10", "trt10-parser-refitter-model-buffer.manifest.json");
        string trt11 = ReadTensorRtManifest("v11", "trt11-onnx-model-buffer-lifecycle.manifest.json");

        Assert.Contains("trt10-parser-refitter-refit-from-bytes", trt10);
        Assert.Contains("trt10-parser-refitter-refit-from-file", trt10);
        Assert.Contains("\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10\"", trt10);

        Assert.Contains("trt11-onnx-parser-load-model-proto", trt11);
        Assert.Contains("trt11-onnx-parser-load-initializer", trt11);
        Assert.Contains("trt11-onnx-parser-parse-model-proto", trt11);
        Assert.Contains("trt11-parser-refitter-refit-from-bytes", trt11);
        Assert.Contains("trt11-parser-refitter-refit-from-file", trt11);
        Assert.Contains("trt11-parser-refitter-load-model-proto", trt11);
        Assert.Contains("trt11-parser-refitter-load-initializer", trt11);
        Assert.Contains("trt11-parser-refitter-refit-model-proto", trt11);
        Assert.Contains("\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11\"", trt11);

        Assert.Contains("\"type\": \"const void*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", trt10 + trt11);
        Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\", \"managedType\": \"out int\"", trt10 + trt11);
        Assert.DoesNotContain("-deferred", trt10 + trt11);
        Assert.DoesNotContain("IParserRefitter*", trt10 + trt11);
        Assert.DoesNotContain("IParser*", trt10 + trt11);
    }

    [Fact]
    public void NativeOnnxModelBufferImplementationKeepsPointersCallerOwned()
    {
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string refitterSource = ReadSource("native", "src", "tensorrt", "common", "parser_refitter_diagnostics.inc");
        string trt10Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string trt11Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");

        Assert.Contains("parser_payload->loadModelProto(model_data, model_size, model_path)", trt11Api);
        Assert.Contains("parser_payload->loadInitializer(name, data, data_size)", trt11Api);
        Assert.Contains("parser_payload->parseModelProto()", trt11Api);
        Assert.Contains("mark_onnx_parser_support_ready(parser_payload, false)", trt11Api);

        Assert.Contains("payload->refitFromBytes(model_data, model_size, model_path)", refitterSource);
        Assert.Contains("payload->refitFromFile(file_path)", refitterSource);
        Assert.Contains("payload->loadModelProto(model_data, model_size, model_path)", refitterSource);
        Assert.Contains("payload->loadInitializer(name, data, data_size)", refitterSource);
        Assert.Contains("payload->refitModelProto()", refitterSource);
        Assert.Contains("report_parser_refitter_native_exception", refitterSource);
        Assert.Contains("report_parser_refitter_unknown_exception", refitterSource);

        Assert.Contains("jyppx_trt10_parser_refitter_refit_from_bytes", trt10Header);
        Assert.Contains("jyppx_trt10_parser_refitter_refit_from_file", trt10Header);
        Assert.Contains("jyppx_trt11_onnx_parser_load_model_proto", trt11Header);
        Assert.Contains("jyppx_trt11_onnx_parser_load_initializer", trt11Header);
        Assert.Contains("jyppx_trt11_parser_refitter_load_initializer", trt11Header);
    }

    [Fact]
    public void ManagedOnnxModelBufferWrappersOwnInitializerLifetime()
    {
        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs");
        string refitter = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.cs");
        string pinSet = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "TensorRtPinnedInitializerSet.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxModelBuffer.cs");

        Assert.Contains("private readonly TensorRtPinnedInitializerSet _initializerPins", parser);
        Assert.Contains("public bool LoadModelProto(byte[] modelData, string? modelPath = null)", parser);
        Assert.Contains("public bool LoadInitializer(string name, byte[] data)", parser);
        Assert.Contains("public bool ParseLoadedModel()", parser);
        Assert.Contains("_handle.Dispose();", parser);
        Assert.Contains("_initializerPins.Dispose();", parser);

        Assert.Contains("public bool RefitFromBytes(byte[] modelData, string? modelPath = null)", refitter);
        Assert.Contains("public bool RefitFromFile(string filePath)", refitter);
        Assert.Contains("public bool LoadModelProto(byte[] modelData, string? modelPath = null)", refitter);
        Assert.Contains("public bool LoadInitializer(string name, byte[] data)", refitter);
        Assert.Contains("public bool RefitLoadedModel()", refitter);
        Assert.Contains("_initializerPins.Dispose();", refitter);

        Assert.Contains("Dictionary<string, PinnedInitializer>", pinSet);
        Assert.Contains("GCHandle.Alloc(Data, GCHandleType.Pinned)", pinSet);
        Assert.Contains("Buffer.BlockCopy(source, 0, owned, 0, source.Length);", pinSet);
        Assert.Contains("previous.Dispose();", pinSet);
        Assert.Contains("next.Dispose();", pinSet);

        Assert.Contains("GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);", interop);
        Assert.Contains("pinned.Free();", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_refit_from_bytes", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_refit_from_bytes", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported", interop);

        Assert.DoesNotContain("public IntPtr", parser + refitter);
        Assert.DoesNotContain("public nint", parser + refitter);
    }

    [Fact]
    public void UnsafeWeightDescriptorAndOriginalDeferredAuditRowsRemain()
    {
        string trt8Deferred = ReadTensorRtManifest("v8", "trt8-cross-version-eighth-batch-onnx-config-parser-deferred.manifest.json");
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("trt8-parser-parse-with-weight-descriptors-deferred", trt8Deferred);
        Assert.Contains("trt10-parser-parse-with-weight-descriptors-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-refit-from-bytes-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-refit-from-file-deferred", trt10Deferred);

        Assert.Contains("trt11-parser-load-initializer-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-load-model-proto-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-parse-model-proto-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-load-initializer-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-load-model-proto-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-refit-model-proto-deferred", trt11Deferred);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
