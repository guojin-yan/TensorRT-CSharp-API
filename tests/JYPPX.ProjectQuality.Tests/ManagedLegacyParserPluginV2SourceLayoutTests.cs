using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedLegacyParserPluginV2SourceLayoutTests
{
    private const string LegacyOriginalNormalizedSha256 =
        "ffeebb61964d50749995b85492645ca0da825dfc10c0e0141cd41f0fda6b813a";
    private const string PluginV2OriginalNormalizedSha256 =
        "798c6ae9c79f14245036051093ea436dd1a4c3a7c235d1285958e1088d586a66";

    public static TheoryData<string, string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Parsing",
            "TensorRtLegacyUffRequiredVersionSnapshot.cs",
            "TensorRtLegacyUffRequiredVersionSnapshot",
            new[] { "ToString" }
        },
        {
            "Parsing",
            "TensorRtCaffeBinaryProtoSnapshot.cs",
            "TensorRtCaffeBinaryProtoSnapshot",
            new[] { "ToString" }
        },
        {
            "Plugins",
            "TensorRtPluginV2LayerMetadata.cs",
            "TensorRtPluginV2LayerMetadata",
            new[] { "ToString" }
        },
        {
            "Plugins",
            "TensorRtLayer.PluginV2Metadata.cs",
            "TensorRtLayer",
            new[]
            {
                "GetPluginV2Metadata", "GetPluginV2LegacyOutputDimensions", "GetPluginV2LegacyWorkspaceSize",
                "SupportsPluginV2LegacyFormat", "GetPluginV2OutputDataType",
                "CanPluginV2BroadcastInputAcrossBatch", "IsPluginV2OutputBroadcastAcrossBatch",
                "TryGetPluginV2Metadata", "IsPluginV2MetadataProbeException", "EnsurePluginV2OwnerLease"
            }
        }
    };

    public static TheoryData<string, string, string[]> PublicProperties => new()
    {
        {
            "Parsing",
            "TensorRtLegacyUffRequiredVersionSnapshot.cs",
            new[]
            {
                "Line", "Major", "Minor", "Patch", "Version", "RetainsNativeParser",
                "CallsProcessGlobalProtobufShutdown", "PointerFreeCopiedMetadata"
            }
        },
        {
            "Parsing",
            "TensorRtCaffeBinaryProtoSnapshot.cs",
            new[]
            {
                "Line", "FileName", "Dimensions", "DataType", "DataLength", "Data", "RetainsNativeObject",
                "PointerFreeCopiedData", "CallsProcessGlobalProtobufShutdown"
            }
        },
        {
            "Plugins",
            "TensorRtPluginV2LayerMetadata.cs",
            new[]
            {
                "Line", "PluginType", "PluginVersion", "PluginNamespace", "SerializationSize",
                "PackedTensorRtVersion", "OutputCount", "HasExtCapability", "HasIoExtCapability",
                "HasDynamicExtCapability", "PluginApiVersionTag", "TensorRtVersion", "TensorRtMajor",
                "TensorRtMinor", "TensorRtPatch", "IsConsistent"
            }
        }
    };

    public static TheoryData<string, string, string> ConstructorOwners => new()
    {
        { "Parsing", "TensorRtLegacyUffRequiredVersionSnapshot.cs", "TensorRtLegacyUffRequiredVersionSnapshot" },
        { "Parsing", "TensorRtCaffeBinaryProtoSnapshot.cs", "TensorRtCaffeBinaryProtoSnapshot" },
        { "Plugins", "TensorRtPluginV2LayerMetadata.cs", "TensorRtPluginV2LayerMetadata" }
    };

    [Theory]
    [MemberData(nameof(FileOwnersAndMethods))]
    public void FilesOwnExactTopLevelTypesAndMethods(
        string module,
        string fileName,
        string expectedType,
        string[] expectedMethods)
    {
        string source = ReadSource(module, fileName);
        Assert.Equal(new[] { expectedType }, EnumerateTopLevelTypeNames(source));
        Assert.Equal(expectedMethods, EnumerateMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(PublicProperties))]
    public void ModelsKeepExactPublicPropertyOrder(string module, string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(module, fileName)));
    }

    [Theory]
    [MemberData(nameof(ConstructorOwners))]
    public void ModelsKeepOneInternalConstructorWithTheirNamesakeTypes(
        string module,
        string fileName,
        string typeName)
    {
        Assert.Single(Regex.Matches(
            ReadSource(module, fileName),
            $@"^    internal {typeName}\(",
            RegexOptions.Multiline));
    }

    [Fact]
    public void LegacyPluralAndPluginMetadataFilesDoNotRetainSecondaryTypes()
    {
        Assert.False(File.Exists(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Parsing",
            "TensorRtLegacyParserDiagnosticSnapshots.cs")));

        string metadata = ReadSource("Plugins", "TensorRtPluginV2LayerMetadata.cs");
        Assert.DoesNotContain("public sealed partial class TensorRtLayer", metadata, StringComparison.Ordinal);
        Assert.Contains(
            "public sealed partial class TensorRtLayer",
            ReadSource("Plugins", "TensorRtLayer.PluginV2Metadata.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void SplitModelsRemainPointerFreeAndOwnerBound()
    {
        string uff = ReadSource("Parsing", "TensorRtLegacyUffRequiredVersionSnapshot.cs");
        string caffe = ReadSource("Parsing", "TensorRtCaffeBinaryProtoSnapshot.cs");
        string metadata = ReadSource("Plugins", "TensorRtPluginV2LayerMetadata.cs");
        string layer = ReadSource("Plugins", "TensorRtLayer.PluginV2Metadata.cs");
        string combined = string.Concat(uff, caffe, metadata, layer);

        Assert.DoesNotContain("public IntPtr", combined, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", combined, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", combined, StringComparison.Ordinal);
        Assert.Contains("RetainsNativeParser => false", uff, StringComparison.Ordinal);
        Assert.Contains("PointerFreeCopiedMetadata => true", uff, StringComparison.Ordinal);
        Assert.Contains("RetainsNativeObject => false", caffe, StringComparison.Ordinal);
        Assert.Contains("PointerFreeCopiedData => true", caffe, StringComparison.Ordinal);
        Assert.Contains("EnsurePluginV2OwnerLease();", layer, StringComparison.Ordinal);
        Assert.Contains("The raw plugin pointer never crosses the native ABI", layer, StringComparison.Ordinal);
    }

    [Fact]
    public void ReaderAndDirectConsumersUseCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        AssertInOrder(
            reader,
            "TensorRtLegacyUffRequiredVersionSnapshot.cs",
            "TensorRtCaffeBinaryProtoSnapshot.cs");
        AssertInOrder(
            reader,
            "TensorRtPluginV2LayerMetadata.cs",
            "TensorRtLayer.PluginV2Metadata.cs");

        foreach (string consumerName in new[]
                 {
                     "LegacyParserReadonlyDiagnosticsUpliftTests.cs",
                     "PluginV2LayerCapabilityReadonlyUpliftTests.cs",
                     "PluginV2LayerMetadataReadonlyUpliftTests.cs"
                 })
        {
            string consumer = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "tests",
                "JYPPX.ProjectQuality.Tests",
                consumerName));
            Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DocumentationReferencesDedicatedLegacyAndPluginV2Files()
    {
        foreach (string organizationPath in new[]
                 {
                     Path.Combine("docs", "articles", "en", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "source-organization.md")
                 })
        {
            string organization = File.ReadAllText(Path.Combine(RepositoryPaths.Root, organizationPath));
            Assert.Contains("TensorRtLegacyUffRequiredVersionSnapshot.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtCaffeBinaryProtoSnapshot.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtLayer.PluginV2Metadata.cs", organization, StringComparison.Ordinal);
        }

        string legacyArticle = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "trt8-legacy-parser-copied-readonly-diagnostics.md"));
        Assert.Contains("TensorRtLegacyUffRequiredVersionSnapshot.cs", legacyArticle, StringComparison.Ordinal);
        Assert.Contains("TensorRtCaffeBinaryProtoSnapshot.cs", legacyArticle, StringComparison.Ordinal);

        string pluginArticle = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "plugin-serialization-paths.md"));
        Assert.Contains("TensorRtPluginV2LayerMetadata.cs", pluginArticle, StringComparison.Ordinal);
        Assert.Contains("TensorRtLayer.PluginV2Metadata.cs", pluginArticle, StringComparison.Ordinal);
        Assert.Contains("不构成 runtime 或 release proof", pluginArticle, StringComparison.Ordinal);
    }

    [Fact]
    public void LegacyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Parsing", "TensorRtLegacyUffRequiredVersionSnapshot.cs")));
        source.Append('\n');
        source.Append(ReadSegment("Parsing", "TensorRtCaffeBinaryProtoSnapshot.cs", "/// <summary>"));
        Assert.Equal(LegacyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void PluginV2FilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Plugins", "TensorRtPluginV2LayerMetadata.cs")));
        source.Append('\n');
        source.Append(ReadSegment(
            "Plugins",
            "TensorRtLayer.PluginV2Metadata.cs",
            "public sealed partial class TensorRtLayer"));
        Assert.Equal(PluginV2OriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public\s+(?:(?:sealed|static|abstract|readonly|partial)\s+)*(?:class|struct|interface|enum|record)\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumerateMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*(?:public|private|internal)\s+(?!(?:delegate)\b)(?:static\s+)?[^\r\n=]+?\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
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

    private static void AssertInOrder(string source, params string[] markers)
    {
        int previous = -1;
        foreach (string marker in markers)
        {
            int current = source.IndexOf(marker, previous + 1, StringComparison.Ordinal);
            Assert.True(current > previous, $"Expected marker in source-set order: {marker}");
            previous = current;
        }
    }

    private static string ReadSegment(string module, string fileName, string marker)
    {
        string source = Normalize(ReadSource(module, fileName));
        int start = source.IndexOf(marker, StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
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
