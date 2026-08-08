using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedEngineParserFeatureLayoutTests
{
    private const string EngineOriginalNormalizedSha256 =
        "6bf5e8e5cec6c5e164a98cdb3577fb202da2ad1b1dd4dfc400926ba150f552cb";
    private const string ParserOriginalNormalizedSha256 =
        "5630772494e31ac2c4a0a0221018dbf55f980ddda786f00a656dbadf77de3f4a";

    public static TheoryData<string, string[]> EngineFeatureMethods => new()
    {
        {
            "TensorMetadata",
            new[]
            {
                "GetIOTensorInfo",
                "GetIOTensorName",
                "GetTensorIndex",
                "GetTensorDataType",
                "GetTensorShape",
                "GetTensorShape64",
                "GetTensorDimensionExtent64",
                "GetTensorIOMode",
                "GetTensorLocation",
                "IsShapeInferenceIO",
                "GetTensorBytesPerComponent",
                "GetTensorBytesPerComponent",
                "GetTensorComponentsPerElement",
                "GetTensorComponentsPerElement",
                "GetTensorFormat",
                "GetTensorFormat",
                "GetTensorFormatDescription",
                "GetTensorFormatDescription",
                "GetTensorVectorizedDimension",
                "GetTensorVectorizedDimension",
                "GetProfileShape",
                "GetProfileShapeValues",
                "GetProfileShape64",
                "GetProfileShapeDimensionExtent64",
                "GetDeviceMemorySizeForProfile",
                "GetDeviceMemorySizeForProfileV2",
                "IsDebugTensor"
            }
        },
        {
            "BindingReports",
            new[]
            {
                "GetIOTensors",
                "GetTensorBinding",
                "GetTensorBinding",
                "GetBindingReport",
                "GetBindingReport"
            }
        },
        {
            "ExecutionContexts",
            new[] { "CreateExecutionContext", "CreateExecutionContextWithoutDeviceMemory" }
        },
        { "Refit", new[] { "CreateRefitter" } },
        { "Inspection", new[] { "CreateInspector" } }
    };

    public static TheoryData<string, string[]> ParserFeatureMethods => new()
    {
        {
            "ModelParsing",
            new[]
            {
                "ParseFromFile",
                "Parse",
                "ParseWithWeightDescriptors",
                "Parse",
                "Parse",
                "Parse"
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
                "LoadInitializer",
                "LoadInitializer",
                "LoadInitializer",
                "LoadInitializer",
                "ParseLoadedModel"
            }
        },
        {
            "TryParse",
            new[]
            {
                "TryParse",
                "TryParse",
                "TryParseWithWeightDescriptors",
                "TryParse",
                "TryParse"
            }
        },
        {
            "Diagnostics",
            new[]
            {
                "GetError",
                "GetDiagnostic",
                "ClearErrors",
                "GetErrors",
                "GetDiagnostics",
                "GetErrorSummary",
                "GetDiagnosticSummary"
            }
        },
        { "OperatorSupport", new[] { "SupportsOperator", "IsSubgraphSupported" } },
        { "Flags", new[] { "GetFlag", "SetFlag", "ClearFlag" } }
    };

    [Theory]
    [MemberData(nameof(EngineFeatureMethods))]
    public void TensorRtEngineFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Engine", $"TensorRtEngine.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(ParserFeatureMethods))]
    public void TensorRtOnnxParserFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Parsing", $"TensorRtOnnxParser.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Fact]
    public void TensorRtEngineCoreRetainsHandleMetadataAndDisposeOnly()
    {
        string core = ReadSource("Engine", "TensorRtEngine.cs");
        string bindings = ReadSource("Engine", "TensorRtEngine.BindingReports.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("private readonly SafeTensorRtObjectHandle _handle;", core, StringComparison.Ordinal);
        Assert.Contains("internal SafeTensorRtObjectHandle Handle => _handle;", core, StringComparison.Ordinal);
        Assert.Contains("public TensorRtApiLine Line { get; }", core, StringComparison.Ordinal);
        Assert.DoesNotContain("private TensorRtEngineBindingReport CreateBindingReport(", core, StringComparison.Ordinal);
        Assert.DoesNotContain("private TensorRtDims? TryGetProfileShape(", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtEngineBindingReport CreateBindingReport(", bindings, StringComparison.Ordinal);
        Assert.Contains("private TensorRtDims? TryGetProfileShape(", bindings, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtOnnxParserCoreRetainsOwnerLifetimeAndSharedValidation()
    {
        string core = ReadSource("Parsing", "TensorRtOnnxParser.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("private readonly TensorRtLogger? _loggerKeepAlive;", core, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtPinnedInitializerSet _initializerPins", core, StringComparison.Ordinal);
        Assert.Contains("public TensorRtOnnxParserFlags Flags", core, StringComparison.Ordinal);
        Assert.Contains("private void ValidateParserFlags(", core, StringComparison.Ordinal);
        Assert.Contains("private void ValidateParserFlag(", core, StringComparison.Ordinal);
        Assert.Contains("private static byte[] CopyModelSegment(", core, StringComparison.Ordinal);
        Assert.Contains("private static byte[] CopyModelStream(", core, StringComparison.Ordinal);
        Assert.Contains("_initializerPins.Dispose();", core, StringComparison.Ordinal);
        Assert.Contains("_loggerKeepAlive?.DetachBorrower();", core, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtEngineFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Engine", "TensorRtEngine.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtEngine");
        const string disposeMarker =
            "    /// <summary>\n    /// Releases the TensorRT engine handle.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0);

        string bindings = ReadPartialBody(
            "Engine",
            "TensorRtEngine.BindingReports.cs",
            "TensorRtEngine");
        const string helperMarker =
            "    private TensorRtEngineBindingReport CreateBindingReport(";
        int helperStart = bindings.IndexOf(helperMarker, StringComparison.Ordinal);
        Assert.True(helperStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        source.Append(ReadPartialBody(
            "Engine",
            "TensorRtEngine.TensorMetadata.cs",
            "TensorRtEngine"));
        source.Append(bindings[..helperStart]);
        source.Append(ReadPartialBody(
            "Engine",
            "TensorRtEngine.ExecutionContexts.cs",
            "TensorRtEngine"));
        source.Append(ReadPartialBody(
            "Engine",
            "TensorRtEngine.Refit.cs",
            "TensorRtEngine"));
        source.Append(ReadPartialBody(
            "Engine",
            "TensorRtEngine.Inspection.cs",
            "TensorRtEngine"));
        source.Append(coreBody[disposeStart..]);
        source.Append(bindings[helperStart..]);
        source.Append('}');
        source.Append('\n');

        Assert.Equal(EngineOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void TensorRtOnnxParserFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Parsing", "TensorRtOnnxParser.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtOnnxParser");
        const string disposeMarker =
            "    /// <summary>\n    /// Releases the native ONNX parser handle.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0);

        string diagnostics = ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.Diagnostics.cs",
            "TensorRtOnnxParser");
        const string aggregateMarker =
            "    /// <summary>\n    /// Gets all parser errors currently reported by TensorRT.";
        int aggregateStart = diagnostics.IndexOf(aggregateMarker, StringComparison.Ordinal);
        Assert.True(aggregateStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.ModelParsing.cs",
            "TensorRtOnnxParser"));
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.ModelLoading.cs",
            "TensorRtOnnxParser"));
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.TryParse.cs",
            "TensorRtOnnxParser"));
        source.Append(diagnostics[..aggregateStart]);
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.OperatorSupport.cs",
            "TensorRtOnnxParser"));
        source.Append(ReadPartialBody(
            "Parsing",
            "TensorRtOnnxParser.Flags.cs",
            "TensorRtOnnxParser"));
        source.Append(diagnostics[aggregateStart..]);
        source.Append(coreBody[disposeStart..]);
        source.Append('}');
        source.Append('\n');

        Assert.Equal(ParserOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static (string Prefix, string Body) SplitClass(
        string source,
        string typeName)
    {
        int declarationStart = source.IndexOf(
            $"public sealed partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        return (
            source[..(bodyStart + 1)],
            RemoveLeadingLineFeed(source[(bodyStart + 1)..bodyEnd]));
    }

    private static string ReadPartialBody(
        string module,
        string fileName,
        string typeName)
    {
        string source = Normalize(ReadSource(module, fileName));
        return SplitClass(source, typeName).Body;
    }

    private static string RemoveLeadingLineFeed(string value)
    {
        Assert.StartsWith("\n", value, StringComparison.Ordinal);
        return value[1..];
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
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexString(digest).ToLowerInvariant();
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
