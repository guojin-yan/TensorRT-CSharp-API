using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ParserRefitterBoundaryTests
{
    [Fact]
    public void ParserRefitterDiagnosticsManifestsPromoteSafeReadonlyEntries()
    {
        string trt10 = ReadTensorRtManifest("v10", "trt10-parser-refitter-diagnostics.manifest.json");
        string trt11 = ReadTensorRtManifest("v11", "trt11-parser-refitter-diagnostics.manifest.json");

        AssertPromotedDiagnosticsManifest(trt10, "10");
        AssertPromotedDiagnosticsManifest(trt11, "11");

        Assert.DoesNotContain("-deferred", trt10);
        Assert.DoesNotContain("-deferred", trt11);
    }

    [Fact]
    public void UnsafeParserRefitterRowsRemainExplicitDeferredForAudit()
    {
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json");
        string trt11GlobalDeferred = ReadTensorRtManifest("v11", "trt11-forty-fourth-batch-global-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("trt10-parser-refitter-create-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-clear-errors-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-get-error-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-get-nb-errors-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-refit-from-bytes-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-refit-from-file-deferred", trt10Deferred);

        Assert.Contains("trt11-parser-refitter-create-deferred", trt11GlobalDeferred);
        Assert.Contains("trt11-parser-refitter-clear-errors-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-get-error-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-get-nb-errors-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-load-initializer-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-load-model-proto-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-refit-from-bytes-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-refit-from-file-deferred", trt11Deferred);
        Assert.Contains("trt11-parser-refitter-refit-model-proto-deferred", trt11Deferred);
    }

    [Fact]
    public void NativeParserRefitterBridgeCopiesDiagnosticsThroughStableAbi()
    {
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "parser_refitter_diagnostics.inc");
        string trt10Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string trt11Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string trt10Manifest = ReadTensorRtManifest("v10", "trt10-parser-refitter-diagnostics.manifest.json");
        string trt11Manifest = ReadTensorRtManifest("v11", "trt11-parser-refitter-diagnostics.manifest.json");

        Assert.Contains("nvonnxparser::createParserRefitter", nativeSource);
        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER", nativeSource);
        Assert.Contains("copy_string_to_buffer", nativeSource);
        Assert.Contains("JYPPX_TensorRtParserErrorInfo", nativeSource);
        Assert.Contains("fill_parser_refitter_error(parser_error, index, out_error);", nativeSource);
        Assert.Contains("destroy_payload<nvonnxparser::IParserRefitter>", nativeSource);
        Assert.Contains("payload->getNbErrors()", nativeSource);
        Assert.Contains("payload->clearErrors()", nativeSource);

        string publicAbi = string.Join(Environment.NewLine, trt10Header, trt11Header, trt10Manifest, trt11Manifest);
        Assert.DoesNotContain("IParserError", publicAbi);
        Assert.DoesNotContain("IParserError**", publicAbi);
    }

    [Fact]
    public void ManagedWrapperUsesSafeHandleAndCopiedValueObjects()
    {
        string wrapper = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.Diagnostics.cs"));
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.ParserRefitterDiagnostics.cs");
        string refitter = ReadSource("src", "JYPPX.TensorRtSharp", "Refit", "TensorRtRefitter.cs");

        Assert.Contains("public sealed partial class TensorRtOnnxParserRefitter : IDisposable", wrapper);
        Assert.Contains("private readonly SafeTensorRtObjectHandle _handle;", wrapper);
        Assert.Contains("public int ErrorCount", wrapper);
        Assert.Contains("public TensorRtParserErrorInfo GetError", wrapper);
        Assert.Contains("public TensorRtOnnxParserDiagnostic GetDiagnostic", wrapper);
        Assert.Contains("public IReadOnlyList<TensorRtOnnxParserDiagnostic> GetDiagnostics", wrapper);
        Assert.Contains("public TensorRtOnnxParserRefitterDiagnosticSnapshot GetDiagnosticSnapshot()", wrapper);
        Assert.Contains("public void ClearErrors()", wrapper);
        Assert.Contains("AttachBorrower(Line)", wrapper);
        Assert.Contains("DetachBorrower()", wrapper);
        Assert.Contains("Variable-length strings are copied through caller-owned buffers", wrapper);
        Assert.Contains("public sealed class TensorRtOnnxParserRefitterDiagnosticSnapshot", snapshot);
        Assert.Contains("public TensorRtOnnxParserRefitterDiagnosticSummary ToSummary()", snapshot);
        Assert.Contains("public sealed class TensorRtOnnxParserRefitterDiagnosticSummary", snapshot);
        Assert.Contains("public IReadOnlyList<TensorRtOnnxParserDiagnostic> Diagnostics", snapshot);
        Assert.Contains("public string DiagnosticSummary", snapshot);
        Assert.Contains("public int CopiedDiagnosticCount", snapshot);
        Assert.Contains("public int DiagnosticSummaryLength", snapshot);
        Assert.Contains("does not call TensorRT", snapshot);
        Assert.Contains("does not expose native pointers", snapshot);
        Assert.Contains("does not promote the snapshot to runtime or external-model proof", snapshot);
        Assert.DoesNotContain("public IntPtr", wrapper + snapshot);
        Assert.DoesNotContain("public nint", wrapper + snapshot);

        Assert.Contains("public static SafeTensorRtObjectHandle CreateOnnxParserRefitter", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_create", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_create", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported", interop);
        Assert.Contains("NativeTensorRtParserErrorInfo error;", interop);
        Assert.Contains("ReadParserErrorString", interop);
        Assert.Contains("GetOnnxParserRefitterLocalFunctionStack", interop);

        Assert.Contains("public TensorRtOnnxParserRefitter CreateOnnxParserRefitter", refitter);
        Assert.Contains("private int _attachmentCount;", refitter);
        Assert.Contains("_loggerKeepAlive?.DetachBorrower();", refitter);
    }

    [Fact]
    public void BindingGeneratorAndObjectKindKnowParserRefitterHandle()
    {
        string typesHeader = ReadSource("native", "include", "jyppx", "tensorrt", "types.h");
        string objectSource = ReadSource("native", "src", "tensorrt", "common", "object.cpp");
        string generator = ReadSource("tools", "JYPPX.BindingGenerator", "Program.cs");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER", typesHeader);
        Assert.Contains("typedef JYPPX_TensorRtObjectBase JYPPX_TensorRtOnnxParserRefitter;", typesHeader);
        Assert.Contains("\"onnx-parser-refitter\"", objectSource);
        Assert.Contains("\"JYPPX_TensorRtOnnxParserRefitter**\" => moduleSpecific ? \"out SafeTensorRtObjectHandle\" : \"out IntPtr\"", generator);
        Assert.Contains("\"JYPPX_TensorRtOnnxParserRefitter*\" => moduleSpecific ? \"SafeTensorRtObjectHandle\" : \"IntPtr\"", generator);
    }

    private static void AssertPromotedDiagnosticsManifest(string manifest, string line)
    {
        Assert.Contains($"trt{line}-parser-refitter-create", manifest);
        Assert.Contains($"trt{line}-parser-refitter-get-error-count", manifest);
        Assert.Contains($"trt{line}-parser-refitter-get-error", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-description", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-file", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-function", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-node-name", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-node-operator", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-local-function-stack-size", manifest);
        Assert.Contains($"trt{line}-parser-refitter-error-get-local-function-stack-entry", manifest);
        Assert.Contains($"trt{line}-parser-refitter-clear-errors", manifest);
        Assert.Contains($"\"versionGuard\": \"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_TensorRtOnnxParserRefitter*\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_TensorRtParserErrorInfo*\"", manifest);
        Assert.Contains("\"moduleManagedType\": \"out NativeTensorRtParserErrorInfo\"", manifest);
        Assert.Contains("\"type\": \"char*\"", manifest);
        Assert.Contains("\"type\": \"size_t*\"", manifest);
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
