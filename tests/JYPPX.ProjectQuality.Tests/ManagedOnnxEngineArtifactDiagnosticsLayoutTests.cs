using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOnnxEngineArtifactDiagnosticsLayoutTests
{
    private const string ArtifactWriterOriginalNormalizedSha256 =
        "ef0a5dbcb12d1924a2c764fc30c8997adc704f165053e8c11cd9a74ec525a5a7";
    private const string BuildDiagnosticsOriginalNormalizedSha256 =
        "10eff02bfbe95e2189c6a0d6cc5fda3846256c626c97b60a2c12345b3509a0a0";

    public static TheoryData<string, string[], string[]> WriterFeatureMembers => new()
    {
        { "OnnxEngineRuntimeArtifactWriter.cs", new[] { "WriteArtifacts", "ComputeSha256" }, Array.Empty<string>() },
        { "OnnxEngineRuntimeArtifactWriter.Times.cs", new[] { "CreateTimesArtifact" }, Array.Empty<string>() },
        { "OnnxEngineRuntimeArtifactWriter.Output.cs", new[] { "CreateOutputArtifact", "ReadFloatSample" }, Array.Empty<string>() },
        { "OnnxEngineRuntimeArtifactWriter.Profile.cs", new[] { "CreateProfileArtifact", "CreateProfileText" }, Array.Empty<string>() },
        {
            "OnnxEngineRuntimeArtifactWriter.EngineReadback.cs",
            new[] { "CreateEngineReadbackArtifact", "GetEngineReadbackArtifactPath", "FirstNonEmpty", "EngineReadbackSkippedReason" },
            Array.Empty<string>()
        },
        {
            "OnnxEngineRuntimeArtifactWriter.RawBindings.cs",
            new[] { "WriteRawBindingsOrBoundary", "CreateRawBindingsManifest", "CreateRawBindingSegments", "CanCaptureRawBindings" },
            new[] { "RawBindingSegment" }
        },
        {
            "OnnxEngineRuntimeArtifactWriter.ProofBoundary.cs",
            new[] { "CreateProofBoundary", "Boundary" },
            new[] { "RuntimeArtifactProofBoundary" }
        },
        { "OnnxEngineRuntimeArtifactWriter.FileIO.cs", new[] { "WriteJson", "WriteText", "WriteBytes" }, Array.Empty<string>() }
    };

    public static TheoryData<string, string[], string[]> DiagnosticsFeatureMembers => new()
    {
        { "OnnxEngineBuildDiagnostics.cs", new[] { "WriteReport" }, Array.Empty<string>() },
        { "OnnxEngineBuildDiagnostics.Json.cs", new[] { "ToJson" }, Array.Empty<string>() },
        { "OnnxEngineBuildDiagnostics.Markdown.cs", new[] { "ToMarkdown" }, Array.Empty<string>() },
        {
            "OnnxEngineBuildDiagnostics.OptionStatus.cs",
            new[]
            {
                "CreateOptionImplementationStatus", "BuildParsedOptions", "BuildAppliedOptions", "BuildParseOnlyOptions",
                "HasAppliedBuilderScalar", "HasAppliedDeploymentControl", "HasAppliedBuildPolicy", "HasNormalizedOption",
                "AddIf", "FormatOptionList"
            },
            Array.Empty<string>()
        },
        { "OnnxEngineBuildDiagnostics.ReportBoundary.cs", new[] { "CreateReportBoundary" }, Array.Empty<string>() }
    };

    public static TheoryData<string, string, string[]> ModelTypeProperties => new()
    {
        {
            "Artifacts", "OnnxEngineRuntimeArtifactData.cs",
            new[]
            {
                "Empty", "TensorName", "Shape", "InputElementCount", "OutputElementCount", "InputPreview", "OutputPreview",
                "ExecutionSummary", "RawOutputBytes", "TimingSamplesMilliseconds", "InputTensors", "OutputTensors",
                "ReferenceValidation", "OutputValidated", "HasOutput", "HasRawOutput"
            }
        },
        {
            "Artifacts", "OnnxEngineRuntimeOutputArtifact.cs",
            new[] { "TensorName", "Shape", "ElementCount", "Preview", "ByteLength", "Sha256" }
        },
        {
            "Build", "OnnxEngineBuildOptionImplementationStatus.cs",
            new[] { "ParsedOptions", "AppliedOptions", "ParseOnlyOptions", "EvidenceBoundary" }
        },
        {
            "Build", "OnnxEngineBuildReportBoundary.cs",
            new[]
            {
                "IsRuntimeProof", "IsBuildOnly", "ForbiddenSubstituteReason", "CopiedDiagnosticsBoundary",
                "ParserDiagnosticsEvidenceKind", "ParserRefitterDiagnosticsEvidenceKind",
                "CanPromoteCopiedDiagnosticsToRuntimeProof", "ParserDiagnosticsOwnerAction", "ForbiddenSubstitutes"
            }
        }
    };

    [Theory]
    [MemberData(nameof(WriterFeatureMembers))]
    public void WriterFeaturePartialsOwnExactMethodsAndNestedTypes(
        string fileName,
        string[] expectedMethods,
        string[] expectedNestedTypes)
    {
        string source = ReadSource("Artifacts", fileName);
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
        Assert.Equal(expectedNestedTypes, EnumerateNestedTypeNames(source));
    }

    [Theory]
    [MemberData(nameof(DiagnosticsFeatureMembers))]
    public void DiagnosticsFeaturePartialsOwnExactMethodsAndNestedTypes(
        string fileName,
        string[] expectedMethods,
        string[] expectedNestedTypes)
    {
        string source = ReadSource("Build", fileName);
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
        Assert.Equal(expectedNestedTypes, EnumerateNestedTypeNames(source));
    }

    [Theory]
    [MemberData(nameof(ModelTypeProperties))]
    public void ModelFilesOwnOneTypeAndExactPublicProperties(
        string module,
        string fileName,
        string[] expectedProperties)
    {
        string source = ReadSource(module, fileName);
        Assert.Single(Regex.Matches(source, @"^public\s+sealed\s+class\s+", RegexOptions.Multiline));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void ArtifactWriterFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Artifacts", "OnnxEngineRuntimeArtifactWriter.cs"));
        string output = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.Output.cs", "OnnxEngineRuntimeArtifactWriter");
        string profile = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.Profile.cs", "OnnxEngineRuntimeArtifactWriter");
        string engine = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.EngineReadback.cs", "OnnxEngineRuntimeArtifactWriter");
        string raw = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.RawBindings.cs", "OnnxEngineRuntimeArtifactWriter");
        string proof = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.ProofBoundary.cs", "OnnxEngineRuntimeArtifactWriter");
        string coreBody = ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.cs", "OnnxEngineRuntimeArtifactWriter");

        int coreHash = coreBody.IndexOf("    private static string ComputeSha256", StringComparison.Ordinal);
        int outputRead = output.IndexOf("    private static IReadOnlyList<float> ReadFloatSample", StringComparison.Ordinal);
        int profileText = profile.IndexOf("    private static string CreateProfileText", StringComparison.Ordinal);
        int engineHelpers = engine.IndexOf("    private static string GetEngineReadbackArtifactPath", StringComparison.Ordinal);
        int rawCapture = raw.IndexOf("    private static bool CanCaptureRawBindings", StringComparison.Ordinal);
        int rawType = raw.IndexOf("    private sealed class RawBindingSegment", StringComparison.Ordinal);
        int proofType = proof.IndexOf("    private sealed class RuntimeArtifactProofBoundary", StringComparison.Ordinal);
        Assert.True(coreHash >= 0 && outputRead >= 0 && profileText >= 0 && engineHelpers >= 0);
        Assert.True(rawCapture >= 0 && rawType > rawCapture && proofType >= 0);

        int declaration = core.IndexOf("public static partial class OnnxEngineRuntimeArtifactWriter", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart >= 0);

        StringBuilder source = new();
        source.Append(core[..declaration]);
        source.Append(ReadTopLevelSegment("Artifacts", "OnnxEngineRuntimeArtifactData.cs").TrimEnd('\n'));
        source.Append("\n\n");
        source.Append(ReadTopLevelSegment("Artifacts", "OnnxEngineRuntimeOutputArtifact.cs").TrimEnd('\n'));
        source.Append("\n\n");
        source.Append(core[declaration..(bodyStart + 1)].Replace(
            "public static partial class OnnxEngineRuntimeArtifactWriter",
            "public static class OnnxEngineRuntimeArtifactWriter",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(coreBody[..coreHash]);
        source.Append(ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.Times.cs", "OnnxEngineRuntimeArtifactWriter"));
        source.Append(output[..outputRead]);
        source.Append(coreBody[coreHash..]);
        source.Append(output[outputRead..]);
        source.Append(profile[..profileText]);
        source.Append(engine[..engineHelpers]);
        source.Append(profile[profileText..]);
        source.Append(raw[..rawCapture]);
        source.Append(engine[engineHelpers..]);
        source.Append(raw[rawCapture..rawType]);
        source.Append(proof[..proofType]);
        source.Append(raw[rawType..]);
        source.Append(ReadPartialBody("Artifacts", "OnnxEngineRuntimeArtifactWriter.FileIO.cs", "OnnxEngineRuntimeArtifactWriter"));
        source.Append(proof[proofType..]);
        source.Append("}\n");

        Assert.Equal(ArtifactWriterOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void BuildDiagnosticsFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Build", "OnnxEngineBuildDiagnostics.cs"));
        int declaration = core.IndexOf("public static partial class OnnxEngineBuildDiagnostics", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart >= 0);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public static partial class OnnxEngineBuildDiagnostics",
            "public static class OnnxEngineBuildDiagnostics",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Build", "OnnxEngineBuildDiagnostics.cs", "OnnxEngineBuildDiagnostics"));
        source.Append(ReadPartialBody("Build", "OnnxEngineBuildDiagnostics.Json.cs", "OnnxEngineBuildDiagnostics"));
        source.Append(ReadPartialBody("Build", "OnnxEngineBuildDiagnostics.Markdown.cs", "OnnxEngineBuildDiagnostics"));
        source.Append(ReadPartialBody("Build", "OnnxEngineBuildDiagnostics.OptionStatus.cs", "OnnxEngineBuildDiagnostics"));
        source.Append(ReadPartialBody("Build", "OnnxEngineBuildDiagnostics.ReportBoundary.cs", "OnnxEngineBuildDiagnostics"));
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment("Build", "OnnxEngineBuildOptionImplementationStatus.cs").TrimEnd('\n'));
        source.Append("\n\n");
        source.Append(ReadTopLevelSegment("Build", "OnnxEngineBuildReportBoundary.cs"));

        Assert.Equal(BuildDiagnosticsOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
        int start = source.IndexOf("public sealed class", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
    }

    private static string[] EnumerateMethodNames(string source)
    {
        int nestedType = source.IndexOf("    private sealed class", StringComparison.Ordinal);
        if (nestedType >= 0)
        {
            source = source[..nestedType];
        }

        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?:static\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateNestedTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s+private\s+sealed\s+class\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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
