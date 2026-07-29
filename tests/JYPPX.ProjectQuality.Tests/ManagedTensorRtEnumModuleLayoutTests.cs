using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedTensorRtEnumModuleLayoutTests
{
    private const string OriginalNormalizedSha256 =
        "f369f7a9d557889dee633d8e7c39de77231c9c61ca45a5b35476d95d8e429dff";

    private static readonly string[] OriginalTypeOrder =
    {
        "TensorRtNetworkDefinitionCreationFlags",
        "TensorRtDataType",
        "TensorRtIOMode",
        "TensorRtOnnxParserFlags",
        "TensorRtOnnxParserFlag",
        "TensorRtTensorLocation",
        "TensorRtExecutionContextAllocationStrategy",
        "TensorRtSerializationFlag",
        "TensorRtSerializationFlags",
        "TensorRtTempfileControlFlag",
        "TensorRtTempfileControlFlags",
        "TensorRtTensorFormat",
        "TensorRtTensorFormats",
        "TensorRtEngineCapability",
        "TensorRtPreviewFeature",
        "TensorRtHardwareCompatibilityLevel",
        "TensorRtRuntimePlatform",
        "TensorRtRnnOperation",
        "TensorRtRnnDirection",
        "TensorRtRnnInputMode",
        "TensorRtRnnGateType",
        "TensorRtDeviceType",
        "TensorRtElementWiseOperation",
        "TensorRtMatrixOperation",
        "TensorRtReduceOperation",
        "TensorRtDistributedReduceOperation",
        "TensorRtUnaryOperation",
        "TensorRtActivationType",
        "TensorRtPoolingType",
        "TensorRtScaleMode",
        "TensorRtPaddingMode",
        "TensorRtResizeMode",
        "TensorRtInterpolationMode",
        "TensorRtResizeCoordinateTransformation",
        "TensorRtResizeSelector",
        "TensorRtResizeRoundMode",
        "TensorRtSliceMode",
        "TensorRtSampleMode",
        "TensorRtFillOperation",
        "TensorRtTopKOperation",
        "TensorRtGatherMode",
        "TensorRtScatterMode",
        "TensorRtCumulativeOperation",
        "TensorRtCollectiveOperation",
        "TensorRtLoopOutputKind",
        "TensorRtTripLimitKind",
        "TensorRtEngineStat",
        "TensorRtTilingOptimizationLevel",
        "TensorRtQuantizationFlag",
        "TensorRtQuantizationFlags",
        "TensorRtTacticSources",
        "TensorRtLayerType",
        "TensorRtBoundingBoxFormat",
        "TensorRtOptimizationProfileSelector",
        "TensorRtLayerInformationFormat",
        "TensorRtKvCacheMode",
        "TensorRtAttentionIoForm",
        "TensorRtAttentionNormalizationOperation",
        "TensorRtCausalMaskKind",
        "TensorRtMoEActivationType",
        "TensorRtProfilingVerbosity",
        "TensorRtBuilderFlag",
        "TensorRtBuilderFlags",
        "TensorRtMemoryPoolType"
    };

    public static TheoryData<string, string, string[], string[]> ModuleEnumOwners => new()
    {
        {
            "Core", "TensorRtTensorCoreEnums.cs",
            new[] { "TensorRtDataType", "TensorRtIOMode", "TensorRtTensorLocation" },
            Array.Empty<string>()
        },
        {
            "Network", "TensorRtNetworkEnums.cs",
            new[] { "TensorRtNetworkDefinitionCreationFlags", "TensorRtTensorFormat", "TensorRtTensorFormats" },
            new[] { "TensorRtNetworkDefinitionCreationFlags", "TensorRtTensorFormats" }
        },
        {
            "Parsing", "TensorRtOnnxParserEnums.cs",
            new[] { "TensorRtOnnxParserFlags", "TensorRtOnnxParserFlag" },
            new[] { "TensorRtOnnxParserFlags" }
        },
        {
            "Execution", "TensorRtExecutionEnums.cs",
            new[] { "TensorRtExecutionContextAllocationStrategy" },
            Array.Empty<string>()
        },
        {
            "Serialization", "TensorRtSerializationEnums.cs",
            new[] { "TensorRtSerializationFlag", "TensorRtSerializationFlags", "TensorRtTempfileControlFlag", "TensorRtTempfileControlFlags" },
            new[] { "TensorRtSerializationFlags", "TensorRtTempfileControlFlags" }
        },
        {
            "Engine", "TensorRtEngineEnums.cs",
            new[]
            {
                "TensorRtEngineCapability", "TensorRtHardwareCompatibilityLevel", "TensorRtEngineStat",
                "TensorRtLayerInformationFormat", "TensorRtKvCacheMode", "TensorRtProfilingVerbosity"
            },
            Array.Empty<string>()
        },
        {
            "Runtime", "TensorRtRuntimeEnums.cs",
            new[] { "TensorRtRuntimePlatform" },
            Array.Empty<string>()
        },
        {
            "Builder", "TensorRtBuilderEnums.cs",
            new[]
            {
                "TensorRtPreviewFeature", "TensorRtDeviceType", "TensorRtTilingOptimizationLevel", "TensorRtQuantizationFlag",
                "TensorRtQuantizationFlags", "TensorRtTacticSources", "TensorRtBuilderFlag", "TensorRtBuilderFlags",
                "TensorRtMemoryPoolType"
            },
            new[] { "TensorRtQuantizationFlags", "TensorRtTacticSources", "TensorRtBuilderFlags" }
        },
        {
            "Layers", "TensorRtRnnEnums.cs",
            new[] { "TensorRtRnnOperation", "TensorRtRnnDirection", "TensorRtRnnInputMode", "TensorRtRnnGateType" },
            Array.Empty<string>()
        },
        {
            "Layers", "TensorRtLayerOperationEnums.cs",
            new[]
            {
                "TensorRtElementWiseOperation", "TensorRtMatrixOperation", "TensorRtReduceOperation",
                "TensorRtDistributedReduceOperation", "TensorRtUnaryOperation", "TensorRtActivationType", "TensorRtPoolingType",
                "TensorRtScaleMode", "TensorRtPaddingMode", "TensorRtSliceMode", "TensorRtSampleMode", "TensorRtFillOperation",
                "TensorRtTopKOperation", "TensorRtGatherMode", "TensorRtScatterMode", "TensorRtCumulativeOperation",
                "TensorRtCollectiveOperation"
            },
            Array.Empty<string>()
        },
        {
            "Layers", "TensorRtResizeEnums.cs",
            new[]
            {
                "TensorRtResizeMode", "TensorRtInterpolationMode", "TensorRtResizeCoordinateTransformation",
                "TensorRtResizeSelector", "TensorRtResizeRoundMode"
            },
            Array.Empty<string>()
        },
        {
            "Layers", "TensorRtLayerMetadataEnums.cs",
            new[] { "TensorRtLayerType", "TensorRtBoundingBoxFormat" },
            Array.Empty<string>()
        },
        {
            "Layers", "TensorRtAttentionEnums.cs",
            new[]
            {
                "TensorRtAttentionIoForm", "TensorRtAttentionNormalizationOperation", "TensorRtCausalMaskKind",
                "TensorRtMoEActivationType"
            },
            Array.Empty<string>()
        },
        {
            "ControlFlow", "TensorRtControlFlowEnums.cs",
            new[] { "TensorRtLoopOutputKind", "TensorRtTripLimitKind" },
            Array.Empty<string>()
        },
        {
            "Profiles", "TensorRtOptimizationProfileEnums.cs",
            new[] { "TensorRtOptimizationProfileSelector" },
            Array.Empty<string>()
        }
    };

    [Theory]
    [MemberData(nameof(ModuleEnumOwners))]
    public void ModuleFilesOwnExactEnumsAndFlags(
        string module,
        string fileName,
        string[] expectedEnums,
        string[] expectedFlagsEnums)
    {
        string source = ReadSource(module, fileName);
        Assert.Equal(expectedEnums, EnumerateEnumNames(source));
        Assert.Equal(expectedFlagsEnums, EnumerateFlagsEnumNames(source));
    }

    [Fact]
    public void AggregateCoreFileIsRemovedAndAllEnumsAreUnique()
    {
        Assert.False(File.Exists(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Core",
            "TensorRtEnums.cs")));

        string[] actual = ModuleEnumOwners
            .SelectMany(row => EnumerateEnumNames(ReadSource((string)row[0], (string)row[1])))
            .ToArray();
        Assert.Equal(64, actual.Length);
        Assert.Equal(64, actual.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(OriginalTypeOrder.OrderBy(static name => name), actual.OrderBy(static name => name));
    }

    [Fact]
    public void ExplicitUnderlyingTypesRemainExact()
    {
        string[] expectedUInt32Enums =
        {
            "TensorRtNetworkDefinitionCreationFlags",
            "TensorRtOnnxParserFlags",
            "TensorRtSerializationFlags",
            "TensorRtTempfileControlFlags",
            "TensorRtTensorFormats",
            "TensorRtQuantizationFlags",
            "TensorRtTacticSources",
            "TensorRtBuilderFlags"
        };

        List<string> actual = new();
        foreach (object[] row in ModuleEnumOwners)
        {
            string source = ReadSource((string)row[0], (string)row[1]);
            actual.AddRange(Regex.Matches(
                    source,
                    @"^public\s+enum\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*uint\s*$",
                    RegexOptions.Multiline)
                .Select(match => match.Groups["name"].Value));
        }

        Assert.Equal(expectedUInt32Enums.OrderBy(static name => name), actual.OrderBy(static name => name));
    }

    [Fact]
    public void ModuleFilesRecomposeTheOriginalSource()
    {
        Dictionary<string, string> blocks = new(StringComparer.Ordinal);
        string? header = null;
        foreach (object[] row in ModuleEnumOwners)
        {
            string source = Normalize(ReadSource((string)row[0], (string)row[1]));
            int firstBlock = source.IndexOf("/// <summary>", StringComparison.Ordinal);
            Assert.True(firstBlock >= 0);
            header ??= source[..firstBlock];

            MatchCollection matches = Regex.Matches(
                source,
                @"^public\s+enum\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
                RegexOptions.Multiline);
            for (int index = 0; index < matches.Count; index++)
            {
                Match match = matches[index];
                int start = source.LastIndexOf("/// <summary>", match.Index, StringComparison.Ordinal);
                int end = index + 1 < matches.Count
                    ? source.LastIndexOf("/// <summary>", matches[index + 1].Index, StringComparison.Ordinal)
                    : source.Length;
                Assert.True(start >= 0 && end > start);
                blocks.Add(match.Groups["name"].Value, source[start..end]);
            }
        }

        StringBuilder sourceBuilder = new(header);
        for (int index = 0; index < OriginalTypeOrder.Length; index++)
        {
            sourceBuilder.Append(blocks[OriginalTypeOrder[index]].TrimEnd('\n'));
            sourceBuilder.Append(index + 1 < OriginalTypeOrder.Length ? "\n\n" : "\n");
        }

        Assert.Equal(OriginalNormalizedSha256, ComputeSha256(sourceBuilder.ToString()));
    }

    private static string[] EnumerateEnumNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public\s+enum\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateFlagsEnumNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"\[Flags\]\npublic\s+enum\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}
