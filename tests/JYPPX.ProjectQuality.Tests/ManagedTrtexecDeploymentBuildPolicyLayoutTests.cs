using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedTrtexecDeploymentBuildPolicyLayoutTests
{
    private const string DeploymentOriginalNormalizedSha256 =
        "cd3154c7ed1bc1fc75023dead6f623b460378ed25898766c7f4bbebf50432e43";
    private const string BuildPolicyOriginalNormalizedSha256 =
        "043a44f417ce162464aa9d9266690034df473d373157cde2cc58a644fef77d47";

    public static TheoryData<string, string[]> DeploymentFileMethods => new()
    {
        { "TrtexecLikeDeploymentOptions.cs", Array.Empty<string>() },
        { "TrtexecLikeDeploymentOptions.Arguments.cs", new[] { "ToArgumentSegments" } },
        { "TrtexecLikeDeploymentOptions.Diagnostics.cs", new[] { "ToDiagnostics" } },
        { "TrtexecLikeDeploymentOptions.Tactics.cs", new[] { "ResolveTacticSources" } },
        { "TrtexecLikeDeploymentOptions.MemoryPools.cs", new[] { "MemoryPoolSizesToArgument" } },
        {
            "TrtexecLikeDeploymentOptions.ProjectionHelpers.cs",
            new[]
            {
                "AddDiagnostic", "AddDiagnostic", "AddDiagnostic", "AddDiagnostic", "Add", "AddSwitch",
                "QuoteIfNeeded", "FormatNullable", "FormatBytesMiB", "FormatBytes"
            }
        },
        { "TrtexecLikeMemoryPoolSize.cs", new[] { "ToTensorRtMemoryPoolType", "ToArgumentSegment" } }
    };

    public static TheoryData<string, string[]> BuildPolicyFileMethods => new()
    {
        {
            "TrtexecLikeBuildPolicy.cs",
            new[]
            {
                "NormalizeIoFormats", "NormalizePrecisionConstraints", "NormalizeLayerPrecisions",
                "NormalizeLayerOutputTypes", "ValidatePolicyCombination", "Apply"
            }
        },
        { "TrtexecLikeBuildPolicy.Parsing.cs", new[] { "ParseIoFormats", "ParseLayerRules", "ResolveLayerRule" } },
        { "TrtexecLikeBuildPolicy.IoFormats.cs", new[] { "ApplyIoFormats" } },
        { "TrtexecLikeBuildPolicy.Precision.cs", new[] { "ApplyPrecisionConstraints" } },
        { "TrtexecLikeBuildPolicy.Layers.cs", new[] { "ApplyLayerPolicies" } },
        { "TrtexecLikeBuildPolicy.Rules.cs", new[] { "TrackMatchingRules", "EnsureEveryRuleMatched", "IsDataTypeSupported" } },
        { "TrtexecLikeBuildPolicy.DataTypes.cs", new[] { "ParseDataType", "ParseTensorFormat" } },
        { "TrtexecLikeIoFormatSpec.cs", new[] { "ToString" } },
        { "TrtexecLikeLayerTypeRule.cs", new[] { "Matches", "ToString" } }
    };

    public static TheoryData<string, string[]> ModelProperties => new()
    {
        {
            "TrtexecLikeDeploymentOptions.cs",
            new[]
            {
                "Default", "DeviceOrdinal", "BuilderOptimizationLevel", "MaxAuxStreams", "DlaCore", "AllowGpuFallback",
                "TacticSources", "MemoryPoolSizes", "InputIOFormats", "OutputIOFormats", "CalibrationCacheFile", "DirectIO",
                "Sparsity", "StronglyTyped", "MinTiming", "AvgTiming", "PrecisionConstraints", "LayerPrecisions",
                "LayerOutputTypes", "Fp8", "Best", "DumpRefit", "AllowWeightStreaming", "MarkDebug", "DumpDebugTensors",
                "VersionCompatible", "ExcludeLeanRuntime", "StripWeights", "Refit", "WeightStreamingBudgetBytes",
                "WeightStreamingBudget", "ExportTimingCachePath", "Safe", "Consistency", "BuilderCache", "NoBuilderCache",
                "MaxNbTactics", "TilingOptimizationLevel", "L2LimitForTilingBytes", "QuantizationFlags", "RefitFromOnnxPath",
                "SaveRefittedEnginePath"
            }
        },
        { "TrtexecLikeMemoryPoolSize.cs", new[] { "Name", "SizeMiB", "SizeBytes" } },
        { "TrtexecLikeIoFormatSpec.cs", new[] { "DataTypeToken", "DataType", "FormatText", "Formats" } },
        { "TrtexecLikeLayerTypeRule.cs", new[] { "Pattern", "DataTypeTokens", "DataTypes", "HasWildcard" } }
    };

    [Theory]
    [MemberData(nameof(DeploymentFileMethods))]
    public void DeploymentFilesOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(BuildPolicyFileMethods))]
    public void BuildPolicyFilesOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(ModelProperties))]
    public void ModelFilesOwnExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void DeploymentFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("TrtexecLikeDeploymentOptions.cs"));
        int declaration = core.IndexOf("public sealed partial class TrtexecLikeDeploymentOptions", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public sealed partial class TrtexecLikeDeploymentOptions",
            "public sealed class TrtexecLikeDeploymentOptions",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.cs", "TrtexecLikeDeploymentOptions"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.Arguments.cs", "TrtexecLikeDeploymentOptions"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.Diagnostics.cs", "TrtexecLikeDeploymentOptions"));
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.Tactics.cs", "TrtexecLikeDeploymentOptions"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.MemoryPools.cs", "TrtexecLikeDeploymentOptions"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeDeploymentOptions.ProjectionHelpers.cs", "TrtexecLikeDeploymentOptions"));
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment("TrtexecLikeMemoryPoolSize.cs", "public sealed class"));

        Assert.Equal(DeploymentOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void BuildPolicyFilesRecomposeTheOriginalSource()
    {
        string ioFormat = Normalize(ReadSource("TrtexecLikeIoFormatSpec.cs"));
        int firstType = ioFormat.IndexOf("internal sealed class", StringComparison.Ordinal);
        Assert.True(firstType >= 0);

        StringBuilder source = new(ioFormat[..firstType]);
        source.Append(ioFormat[firstType..].TrimEnd('\n'));
        source.Append("\n\n");
        source.Append(ReadTopLevelSegment("TrtexecLikeLayerTypeRule.cs", "internal sealed class").TrimEnd('\n'));
        source.Append("\n\n");

        string core = Normalize(ReadSource("TrtexecLikeBuildPolicy.cs"));
        int declaration = core.IndexOf("internal static partial class TrtexecLikeBuildPolicy", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);
        source.Append(core[declaration..(bodyStart + 1)].Replace(
            "internal static partial class TrtexecLikeBuildPolicy",
            "internal static class TrtexecLikeBuildPolicy",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.Parsing.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.IoFormats.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.Precision.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.Layers.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.Rules.cs", "TrtexecLikeBuildPolicy"));
        source.Append('\n');
        source.Append(ReadPartialBody("TrtexecLikeBuildPolicy.DataTypes.cs", "TrtexecLikeBuildPolicy"));
        source.Append("}\n");

        Assert.Equal(BuildPolicyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadPartialBody(string fileName, string typeName)
    {
        string source = Normalize(ReadSource(fileName));
        int declaration = source.IndexOf(typeName, StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declaration);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declaration >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static string ReadTopLevelSegment(string fileName, string marker)
    {
        string source = Normalize(ReadSource(fileName));
        int start = source.IndexOf(marker, StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
    }

    private static string[] EnumerateMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?:(?:static|override)\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
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

    private static string ReadSource(string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Trtexec",
            fileName));
    }
}
