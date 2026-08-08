using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedMnistRuntimeParserLayoutTests
{
    private const string MnistOriginalNormalizedSha256 =
        "158263db48c5ec9de64003d2cb327e44e45e0d5ac612c050d63854aa43e9474f";
    private const string ParserOriginalNormalizedSha256 =
        "65456303abb1f56284a43e8ea64224a35ba7da41af9653cc7fd2222a3f077b50";

    private const string MnistOriginalHeader =
        "using System;\n" +
        "using System.Collections.Generic;\n" +
        "using System.Globalization;\n" +
        "using System.IO;\n" +
        "using System.Linq;\n" +
        "using System.Runtime.InteropServices;\n" +
        "using System.Security.Cryptography;\n" +
        "using System.Text;\n" +
        "using System.Text.Json;\n" +
        "using JYPPX.CudaSharp;\n" +
        "using JYPPX.TensorRtSharp.Shared.Interop;\n" +
        "using JYPPX.TensorRtSharp;\n\n" +
        "namespace JYPPX.TensorRtSharp.Tools;\n\n";

    public static TheoryData<string, string[]> MnistFileMethods => new()
    {
        { "MnistOnnxRuntimeOptions.cs", new[] { "ToCommandLine", "Quote" } },
        {
            "MnistPgmReader.cs",
            new[] { "Read", "ToTensorInput", "ReadToken", "SkipWhitespaceAndComments", "ConsumePixelSeparator", "IsWhitespace", "ParsePositiveInt" }
        },
        { "MnistOutputClassifier.cs", new[] { "Classify" } },
        { "MnistRuntimeEnvironment.cs", new[] { "Capture" } },
        { "MnistOnnxRuntimeResult.cs", new[] { "HashText" } },
        { "MnistOnnxRuntimeService.cs", new[] { "Execute" } },
        { "MnistOnnxRuntimeService.Artifacts.cs", new[] { "ComputeSha256", "ComputeFileSha256", "ToBytes" } },
        { "MnistOnnxRuntimeService.Tensors.cs", new[] { "SingleTensor", "EnsureFloatDeviceTensor", "ResolveShape", "IsConcrete", "ElementCount" } },
        { "MnistOnnxRuntimeService.Validation.cs", new[] { "ValidateOptions" } },
        { "MnistOnnxRuntimeDiagnostics.cs", new[] { "WriteArtifacts", "WriteJson", "WriteText", "WriteBytes" } }
    };

    public static TheoryData<string, string[]> ParserFileMethods => new()
    {
        { "TrtexecLikeParser.cs", new[] { "Parse" } },
        {
            "TrtexecLikeParser.Arguments.cs",
            new[]
            {
                "GetValue", "HasSwitch", "ParseThreadMode", "FullPathOrEmpty", "PathsEqual", "ParseList",
                "ParsePluginLibraries", "AddValues", "IsOptionName", "ShouldTreatEngineAliasAsLoad", "FirstNonEmpty"
            }
        },
        {
            "TrtexecLikeParser.ScalarParsing.cs",
            new[]
            {
                "ParsePositiveInt", "ParseNonNegativeInt", "ParseOptionalNonNegativeInt", "ParseOptionalPositiveInt",
                "ParseOptionalRangeFloat", "ParseReferenceNaNPolicy", "ParseReferenceInfinityPolicy", "ParseRangeInt"
            }
        },
        {
            "TrtexecLikeParser.BuildOptionValues.cs",
            new[]
            {
                "ParseWorkspaceBytes", "ParseOptionalMemorySizeBytes", "ParseOptionalLongMemorySizeBytes",
                "ParseOptionalTilingOptimizationLevel", "ParseOptionalQuantizationFlags", "ParseMemoryPoolSizes",
                "NormalizeProfilingVerbosity", "NormalizeTacticSources", "NormalizeSparsity"
            }
        },
        { "TrtexecLikeParser.MemoryUnits.cs", new[] { "ParseMemorySizeMiB", "ParseMemorySizeBytes", "TryTrimSuffix" } }
    };

    public static TheoryData<string, string[]> MnistModelProperties => new()
    {
        {
            "MnistOnnxRuntimeOptions.cs",
            new[]
            {
                "TensorRtLine", "OnnxPath", "InputPgmPath", "ExpectedDigit", "SaveEnginePath", "ExportReportPath",
                "ExportOutputPath", "ExportPreprocessedInputPath", "WorkspaceBytes", "MinimumConfidence"
            }
        },
        { "MnistPgmImage.cs", new[] { "Width", "Height", "MaxValue", "Pixels" } },
        { "MnistClassification.cs", new[] { "Probabilities", "PredictedDigit", "Confidence" } },
        {
            "MnistRuntimeEnvironment.cs",
            new[]
            {
                "HostOs", "ProcessArchitecture", "MachineName", "GpuName", "ComputeCapability", "GpuMemoryBytes",
                "CudaDriverVersion", "CudaRuntimeVersion", "CudaToolkitVersion", "TensorRtVersion", "BridgeVersion"
            }
        },
        {
            "MnistOnnxRuntimeResult.cs",
            new[]
            {
                "Success", "Skipped", "State", "TensorRtLine", "ModelPath", "InputPath", "EnginePath", "ModelSha256",
                "InputSha256", "PreprocessedInputSha256", "EngineSha256", "Parsed", "EngineSaved", "EngineFileRoundTrip",
                "InferenceRan", "OutputMatch", "ExpectedDigit", "PredictedDigit", "Confidence", "MinimumConfidence",
                "InputTensorName", "InputShape", "InputDataType", "OutputTensorName", "OutputShape", "OutputDataType",
                "Logits", "Probabilities", "ElapsedMilliseconds", "SkipReason", "NormalizedCommandLine",
                "NormalizedCommandSha256", "Environment", "LogLines", "ProofClassification", "IsRealModelRuntimeProof",
                "IsPackageConsumerRuntimeProof", "CanPublishPublicly", "CanCloseReleaseIssue", "ProofBoundary"
            }
        }
    };

    [Theory]
    [MemberData(nameof(MnistFileMethods))]
    public void MnistFilesOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource("Runtime", fileName)));
    }

    [Theory]
    [MemberData(nameof(ParserFileMethods))]
    public void ParserPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource("Trtexec", fileName)));
    }

    [Theory]
    [MemberData(nameof(MnistModelProperties))]
    public void MnistModelFilesOwnOneTypeAndExactPublicProperties(string fileName, string[] expectedProperties)
    {
        string source = ReadSource("Runtime", fileName);
        Assert.Single(Regex.Matches(source, @"^public\s+sealed\s+class\s+", RegexOptions.Multiline));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void LegacyMnistAggregateIsRemoved()
    {
        Assert.False(File.Exists(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Runtime",
            "MnistOnnxRuntime.cs")));
    }

    [Fact]
    public void MnistFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(MnistOriginalHeader);
        AppendTopLevel(source, "MnistOnnxRuntimeOptions.cs");
        AppendTopLevel(source, "MnistPgmImage.cs");
        AppendTopLevel(source, "MnistPgmReader.cs");
        AppendTopLevel(source, "MnistClassification.cs");
        AppendTopLevel(source, "MnistOutputClassifier.cs");
        AppendTopLevel(source, "MnistRuntimeEnvironment.cs");
        AppendTopLevel(source, "MnistOnnxRuntimeResult.cs");

        string core = Normalize(ReadSource("Runtime", "MnistOnnxRuntimeService.cs"));
        int summary = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class MnistOnnxRuntimeService", summary, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(summary >= 0 && declaration > summary && bodyStart > declaration);
        source.Append(core[summary..(bodyStart + 1)].Replace(
            "public sealed partial class MnistOnnxRuntimeService",
            "public sealed class MnistOnnxRuntimeService",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Runtime", "MnistOnnxRuntimeService.cs", "MnistOnnxRuntimeService"));
        source.Append('\n');
        source.Append(ReadPartialBody("Runtime", "MnistOnnxRuntimeService.Artifacts.cs", "MnistOnnxRuntimeService"));
        source.Append('\n');
        source.Append(ReadPartialBody("Runtime", "MnistOnnxRuntimeService.Tensors.cs", "MnistOnnxRuntimeService"));
        source.Append('\n');
        source.Append(ReadPartialBody("Runtime", "MnistOnnxRuntimeService.Validation.cs", "MnistOnnxRuntimeService"));
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment("Runtime", "MnistOnnxRuntimeDiagnostics.cs"));

        Assert.Equal(MnistOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void ParserPartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Trtexec", "TrtexecLikeParser.cs"));
        int declaration = core.IndexOf("public static partial class TrtexecLikeParser", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);

        string arguments = ReadPartialBody("Trtexec", "TrtexecLikeParser.Arguments.cs", "TrtexecLikeParser");
        int firstNonEmpty = arguments.IndexOf("    private static string FirstNonEmpty", StringComparison.Ordinal);
        Assert.True(firstNonEmpty >= 0);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public static partial class TrtexecLikeParser",
            "public static class TrtexecLikeParser",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Trtexec", "TrtexecLikeParser.cs", "TrtexecLikeParser"));
        source.Append('\n');
        source.Append(arguments[..firstNonEmpty]);
        source.Append(ReadPartialBody("Trtexec", "TrtexecLikeParser.ScalarParsing.cs", "TrtexecLikeParser"));
        source.Append('\n');
        source.Append(ReadPartialBody("Trtexec", "TrtexecLikeParser.BuildOptionValues.cs", "TrtexecLikeParser"));
        source.Append('\n');
        source.Append(ReadPartialBody("Trtexec", "TrtexecLikeParser.MemoryUnits.cs", "TrtexecLikeParser"));
        source.Append('\n');
        source.Append(arguments[firstNonEmpty..]);
        source.Append("}\n");

        Assert.Equal(ParserOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static void AppendTopLevel(StringBuilder source, string fileName)
    {
        source.Append(ReadTopLevelSegment("Runtime", fileName).TrimEnd('\n'));
        source.Append("\n\n");
    }

    private static string ReadPartialBody(string module, string fileName, string typeName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int declaration = source.IndexOf(typeName, StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declaration);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declaration >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static string ReadTopLevelSegment(string module, string fileName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int start = source.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
    }

    private static string[] EnumerateMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?:static\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicPropertyNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*public\s+(?:static\s+)?[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|\{)",
                RegexOptions.Multiline)
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
            "JYPPX.TensorRtSharp.Tools",
            module,
            fileName));
    }
}
