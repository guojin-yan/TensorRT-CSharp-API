using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedParserRefitterInferenceFeatureLayoutTests
{
    private const string ParserRefitterOriginalNormalizedSha256 =
        "f18a67b5f9660b0dde1f71288dffd177a18662fc4d7e16455c9341e27e73c35d";
    private const string InferenceBindingsOriginalNormalizedSha256 =
        "96c4de33359c562e7c13d79a5fb0fde4e5c65d2aa0c8fb4eaf5f432aaf1ab288";

    public static TheoryData<string, string[]> ParserRefitterFeatureMethods => new()
    {
        {
            "Refitting",
            new[]
            {
                "RefitFromBytes",
                "RefitFromBytes",
                "RefitFromBytes",
                "RefitFromBytes",
                "RefitFromFile"
            }
        },
        {
            "ModelLoading",
            new[]
            {
                "LoadModelProto",
                "LoadModelProto",
                "LoadModelProto",
                "LoadModelProto",
                "RefitLoadedModel"
            }
        },
        {
            "Initializers",
            new[]
            {
                "LoadInitializer",
                "LoadInitializer",
                "LoadInitializer",
                "LoadInitializer"
            }
        },
        {
            "Diagnostics",
            new[]
            {
                "GetError",
                "GetDiagnostic",
                "GetErrors",
                "GetDiagnostics",
                "GetDiagnosticSnapshot",
                "ClearErrors",
                "GetDiagnosticSummary"
            }
        }
    };

    public static TheoryData<string, string[]> InferenceFeatureMethods => new()
    {
        { "TensorGeometry", new[] { "SetInputShape" } },
        { "Buffers", new[] { "AllocateDeviceBuffer", "UseDeviceBuffer" } },
        {
            "HostTransfers",
            new[] { "CopyInputFromHost", "CopyInputFromHost", "ReadOutputSingles", "ReadOutputBytes" }
        },
        { "AddressBinding", new[] { "BindTensor", "BindAll" } },
        {
            "Execution",
            new[]
            {
                "GetReadiness",
                "EnqueueAsync",
                "ExecuteV2",
                "ExecuteLegacy",
                "EnqueueV2AndSynchronize"
            }
        },
        { "Diagnostics", new[] { "Describe" } }
    };

    [Theory]
    [MemberData(nameof(ParserRefitterFeatureMethods))]
    public void TensorRtOnnxParserRefitterFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Parsing", $"TensorRtOnnxParserRefitter.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(InferenceFeatureMethods))]
    public void TensorRtInferenceBindingsFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Inference", $"TensorRtInferenceBindings.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Fact]
    public void TensorRtOnnxParserRefitterCoreRetainsOwnerLifetimeAndSharedCopyHelpers()
    {
        string core = ReadSource("Parsing", "TensorRtOnnxParserRefitter.cs");
        string diagnostics = ReadSource("Parsing", "TensorRtOnnxParserRefitter.Diagnostics.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("private readonly SafeTensorRtObjectHandle _handle;", core, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtRefitter _refitterKeepAlive;", core, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtLogger _loggerKeepAlive;", core, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtPinnedInitializerSet _initializerPins", core, StringComparison.Ordinal);
        Assert.Contains("private static byte[] CopyModelSegment(", core, StringComparison.Ordinal);
        Assert.Contains("private static byte[] CopyModelStream(", core, StringComparison.Ordinal);
        Assert.Contains("public int ErrorCount", diagnostics, StringComparison.Ordinal);
        Assert.Contains("private static string BuildDiagnosticSummary(", diagnostics, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtInferenceBindingsHelpersStayWithTheirFeatureOwners()
    {
        string core = ReadSource("Inference", "TensorRtInferenceBindings.cs");
        string geometry = ReadSource("Inference", "TensorRtInferenceBindings.TensorGeometry.cs");
        string buffers = ReadSource("Inference", "TensorRtInferenceBindings.Buffers.cs");
        string execution = ReadSource("Inference", "TensorRtInferenceBindings.Execution.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("private readonly TensorRtEngine _engine;", core, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtExecutionContext _context;", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtEngineTensorBinding GetTensor(", core, StringComparison.Ordinal);
        Assert.Contains("private void RefreshReport(", core, StringComparison.Ordinal);
        Assert.Contains("private void ThrowIfDisposed()", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtDims ResolveRuntimeShape(", geometry, StringComparison.Ordinal);
        Assert.Contains("private static int EstimateTensorByteSize(", geometry, StringComparison.Ordinal);
        Assert.Contains("private TensorRtInferenceBuffer EnsureBuffer(", buffers, StringComparison.Ordinal);
        Assert.Contains("private void RemoveOwnedBuffer(", buffers, StringComparison.Ordinal);
        Assert.Contains("private TensorRtExecutionContextReadiness PrepareForExecution(", execution, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtInferenceBindingsKeepsTypedAndRawHostTransfersExplicit()
    {
        string transfers = ReadSource("Inference", "TensorRtInferenceBindings.HostTransfers.cs");

        Assert.Contains("EnsureSinglePrecisionTensor(tensor, \"copied from a float array\")", transfers, StringComparison.Ordinal);
        Assert.Contains("EnsureSinglePrecisionTensor(buffer.Tensor, \"read as a float array\")", transfers, StringComparison.Ordinal);
        Assert.Contains("public byte[] ReadOutputBytes(string tensorName)", transfers, StringComparison.Ordinal);
        Assert.Contains("buffer.Memory.ToArray(buffer.SizeInBytes)", transfers, StringComparison.Ordinal);
        Assert.Contains("tensor.DataType != TensorRtDataType.Float", transfers, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtOnnxParserRefitterFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Parsing", "TensorRtOnnxParserRefitter.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtOnnxParserRefitter");
        prefix = RemovePartialModifier(prefix, "TensorRtOnnxParserRefitter");

        const string disposeMarker =
            "    /// <summary>\n    /// Releases the native ONNX parser-refitter handle.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0);

        string diagnostics = ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParserRefitter.Diagnostics.cs",
            "TensorRtOnnxParserRefitter");
        const string diagnosticMethodsMarker =
            "    /// <summary>\n    /// Gets one parser-refitter error by index.";
        int diagnosticMethodsStart = diagnostics.IndexOf(diagnosticMethodsMarker, StringComparison.Ordinal);
        Assert.True(diagnosticMethodsStart >= 0);

        string modelLoading = ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParserRefitter.ModelLoading.cs",
            "TensorRtOnnxParserRefitter");
        const string refitLoadedMarker =
            "    /// <summary>\n    /// Refits the model proto previously loaded";
        int refitLoadedStart = modelLoading.IndexOf(refitLoadedMarker, StringComparison.Ordinal);
        Assert.True(refitLoadedStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        source.Append(diagnostics[..diagnosticMethodsStart]);
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParserRefitter.Refitting.cs",
            "TensorRtOnnxParserRefitter"));
        source.Append('\n');
        source.Append(modelLoading[..refitLoadedStart]);
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParserRefitter.Initializers.cs",
            "TensorRtOnnxParserRefitter"));
        source.Append('\n');
        source.Append(modelLoading[refitLoadedStart..]);
        source.Append('\n');
        source.Append(diagnostics[diagnosticMethodsStart..]);
        source.Append('\n');
        source.Append(coreBody[disposeStart..]);
        source.Append('}');
        source.Append('\n');

        Assert.Equal(ParserRefitterOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void TensorRtInferenceBindingsFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Inference", "TensorRtInferenceBindings.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtInferenceBindings");
        prefix = RemovePartialModifier(prefix, "TensorRtInferenceBindings");

        const string disposeMarker =
            "    /// <summary>\n    /// Releases CUDA buffers owned by this binding set.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        int getTensorStart = coreBody.IndexOf(
            "    private TensorRtEngineTensorBinding GetTensor(",
            StringComparison.Ordinal);
        int refreshStart = coreBody.IndexOf("    private void RefreshReport(", StringComparison.Ordinal);
        int throwStart = coreBody.IndexOf("    private void ThrowIfDisposed()", StringComparison.Ordinal);
        Assert.True(disposeStart >= 0 && getTensorStart > disposeStart);
        Assert.True(refreshStart > getTensorStart && throwStart > refreshStart);

        string geometry = ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.TensorGeometry.cs",
            "TensorRtInferenceBindings");
        int resolveStart = geometry.IndexOf(
            "    private TensorRtDims ResolveRuntimeShape(",
            StringComparison.Ordinal);
        Assert.True(resolveStart >= 0);

        string buffers = ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.Buffers.cs",
            "TensorRtInferenceBindings");
        int ensureStart = buffers.IndexOf(
            "    private TensorRtInferenceBuffer EnsureBuffer(",
            StringComparison.Ordinal);
        int removeStart = buffers.IndexOf("    private void RemoveOwnedBuffer(", StringComparison.Ordinal);
        Assert.True(ensureStart >= 0 && removeStart > ensureStart);

        string transfers = ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.HostTransfers.cs",
            "TensorRtInferenceBindings");
        const string readOutputMarker =
            "    /// <summary>\n    /// Copies a named output tensor to a single-precision managed array.";
        int readOutputStart = transfers.IndexOf(readOutputMarker, StringComparison.Ordinal);
        Assert.True(readOutputStart >= 0);

        string execution = ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.Execution.cs",
            "TensorRtInferenceBindings");
        int prepareStart = execution.IndexOf(
            "    private TensorRtExecutionContextReadiness PrepareForExecution(",
            StringComparison.Ordinal);
        Assert.True(prepareStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        source.Append(geometry[..resolveStart]);
        source.Append(buffers[..ensureStart]);
        source.Append(transfers[..readOutputStart]);
        source.Append(ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.AddressBinding.cs",
            "TensorRtInferenceBindings"));
        source.Append('\n');
        source.Append(execution[..prepareStart]);
        source.Append(transfers[readOutputStart..]);
        source.Append('\n');
        source.Append(ReadPartialBody(
            "Inference",
            "TensorRtInferenceBindings.Diagnostics.cs",
            "TensorRtInferenceBindings"));
        source.Append('\n');
        source.Append(coreBody[disposeStart..getTensorStart]);
        source.Append(buffers[ensureStart..removeStart]);
        source.Append(coreBody[getTensorStart..refreshStart]);
        source.Append(geometry[resolveStart..]);
        source.Append('\n');
        source.Append(coreBody[refreshStart..throwStart]);
        source.Append(execution[prepareStart..]);
        source.Append('\n');
        source.Append(buffers[removeStart..]);
        source.Append('\n');
        source.Append(coreBody[throwStart..]);
        source.Append('}');
        source.Append('\n');

        Assert.Equal(InferenceBindingsOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static (string Prefix, string Body) SplitClass(string source, string typeName)
    {
        int declarationStart = source.IndexOf(
            $"public sealed partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return (source[..(bodyStart + 1)], body[1..]);
    }

    private static string RemovePartialModifier(string prefix, string typeName)
    {
        return prefix.Replace(
            $"public sealed partial class {typeName}",
            $"public sealed class {typeName}",
            StringComparison.Ordinal);
    }

    private static string ReadPartialBody(string module, string fileName, string typeName)
    {
        return SplitClass(Normalize(ReadSource(module, fileName)), typeName).Body;
    }

    private static string[] EnumeratePublicInstanceMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string ComputeSha256(string value)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)))
            .ToLowerInvariant();
    }

    private static string Normalize(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');
    }

    private static string ReadSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}
