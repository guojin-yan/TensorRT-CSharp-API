using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxParserManagedInputTests
{
    [Fact]
    public void ManagedOnnxParserInputOverloadsCopyIntoManagedByteArrays()
    {
        string parser =
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelParsing.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.TryParse.cs");
        string modelSupport = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelSupport.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.OnnxParserLifecycleAndInput.cs");
        string supportInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.OnnxParserSupport.cs");

        Assert.Contains("public bool Parse(ArraySegment<byte> modelData, string? modelPath = null)", parser);
        Assert.Contains("public bool Parse(ReadOnlySpan<byte> modelData, string? modelPath = null)", parser);
        Assert.Contains("public bool Parse(Stream modelStream, string? modelPath = null)", parser);
        Assert.Contains("public bool TryParse(byte[] modelData, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)", parser);
        Assert.Contains("public bool TryParse(byte[] modelData, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)", parser);
        Assert.Contains("public bool TryParse(Stream modelStream, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)", parser);
        Assert.Contains("public bool TryParse(Stream modelStream, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)", parser);
        Assert.Contains("private static byte[] CopyModelSegment(ArraySegment<byte> modelData, string argumentName)", parser);
        Assert.Contains("Buffer.BlockCopy(modelData.Array, modelData.Offset, buffer, 0, modelData.Count);", parser);
        Assert.Contains("modelStream.CopyTo(copy);", parser);

        Assert.Contains("public TensorRtOnnxModelSupportReport CheckModelSupport(ArraySegment<byte> modelData, string? modelPath = null)", modelSupport);
        Assert.Contains("public TensorRtOnnxModelSupportReport CheckModelSupport(ReadOnlySpan<byte> modelData, string? modelPath = null)", modelSupport);
        Assert.Contains("public TensorRtOnnxModelSupportReport CheckModelSupport(Stream modelStream, string? modelPath = null)", modelSupport);
        Assert.Contains("return CheckModelSupport(CopyModelSegment(modelData, nameof(modelData)), modelPath);", modelSupport);
        Assert.Contains("return CheckModelSupport(CopyModelStream(modelStream, nameof(modelStream)), modelPath);", modelSupport);

        Assert.Contains("GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);", interop);
        Assert.Contains("pinned.Free();", interop);
        Assert.Contains("GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);", supportInterop);
        Assert.Contains("pinned.Free();", supportInterop);
        Assert.DoesNotContain("public IntPtr", parser + modelSupport);
        Assert.DoesNotContain("public nint", parser + modelSupport);
    }

    [Fact]
    public void OnnxToEngineSmokeUsesManagedStreamParserPaths()
    {
        string program = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("using MemoryStream parserModelStream = new MemoryStream(model, writable: false);", program);
        Assert.Contains("bool parsed = parser.Parse(parserModelStream, \"generated-dynamic-identity.onnx\");", program);
        Assert.Contains("TensorRtOnnxModelSupportReport report = supportParser.CheckModelSupport(modelStream, \"generated-dynamic-identity.onnx\");", program);
        Assert.Contains("bool parsed = diagnosticParser.TryParse(invalidModelStream, $\"jyppx-invalid-diagnostic-trt{(int)line}.onnx\", out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics);", program);
        Assert.Contains("ParseStream=", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
