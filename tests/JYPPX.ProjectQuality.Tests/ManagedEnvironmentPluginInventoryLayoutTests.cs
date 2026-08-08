using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedEnvironmentPluginInventoryLayoutTests
{
    private const string EnvironmentProbeOriginalNormalizedSha256 =
        "b5f693d39d2e988ddb85599c714a198246b411abe2207dff688479a5c4f2a3e6";
    private const string PluginInventoryOriginalNormalizedSha256 =
        "1e7b4b8afa5164389039ce85b7d8cb2af17a992c26d999a7a46ba67164a3290b";

    private static readonly string[] EnvironmentFeatureOrder =
    {
        "PluginInitialization",
        "RuntimeMetadata",
        "GlobalPluginRegistry",
        "BuilderPluginRegistry",
        "DependencyProbes",
        "ObjectCreation",
        "BuildChains"
    };

    public static TheoryData<string, string[]> EnvironmentFeatureMethods => new()
    {
        {
            "PluginInitialization",
            new[] { "InitializeBuiltInPlugins", "TryInitializeBuiltInPlugins" }
        },
        {
            "RuntimeMetadata",
            new[] { "GetCurrent", "GetGlobalRuntimeVersion", "TryGetGlobalRuntimeVersion" }
        },
        {
            "GlobalPluginRegistry",
            new[]
            {
                "GetGlobalPluginRegistryInventory",
                "GetGlobalPluginRegistryInventory",
                "TryGetGlobalPluginRegistryInventory",
                "TryGetGlobalPluginRegistryInventory",
                "IsGlobalPluginRegistryParentSearchEnabled",
                "TryIsGlobalPluginRegistryParentSearchEnabled",
                "SetGlobalPluginRegistryParentSearchEnabled",
                "TrySetGlobalPluginRegistryParentSearchEnabled",
                "IsGlobalPluginRegistryAvailable",
                "TryIsGlobalPluginRegistryAvailable",
                "IsGlobalPluginCreatorRegistered",
                "TryIsGlobalPluginCreatorRegistered",
                "TryGetGlobalPluginCreator",
                "TryGetGlobalPluginCreator"
            }
        },
        {
            "BuilderPluginRegistry",
            new[]
            {
                "IsBuilderCapabilityPluginRegistryAvailable",
                "TryIsBuilderCapabilityPluginRegistryAvailable",
                "IsBuilderSafePluginRegistryAvailable",
                "TryIsBuilderSafePluginRegistryAvailable",
                "GetBuilderCapabilityPluginRegistryInventory",
                "GetBuilderCapabilityPluginRegistryInventory",
                "TryGetBuilderCapabilityPluginRegistryInventory",
                "TryGetBuilderCapabilityPluginRegistryInventory",
                "IsBuilderCapabilityPluginCreatorRegistered",
                "TryIsBuilderCapabilityPluginCreatorRegistered",
                "TryGetBuilderCapabilityPluginCreator",
                "TryGetBuilderCapabilityPluginCreator"
            }
        },
        { "DependencyProbes", new[] { "ProbeNativeDependencies", "ProbeRuntime" } },
        {
            "ObjectCreation",
            new[] { "TryCreateLogger", "TryCreateRuntime", "GetRuntimeCreateDiagnostic", "TryCreateBuilder" }
        },
        {
            "BuildChains",
            new[]
            {
                "TryRunTensorRt10MinimalBuildChain",
                "TryBuildTensorRt10SerializedNetworkOnly",
                "TryRunTensorRt8MinimalBuildChain",
                "TryBuildTensorRt8SerializedNetworkOnly",
                "TryRunTensorRt11MinimalBuildChain",
                "TryBuildTensorRt11SerializedNetworkOnly"
            }
        }
    };

    public static TheoryData<string, string[]> PluginTypeFiles => new()
    {
        {
            "TensorRtPluginRegistryTypes.cs",
            new[] { "TensorRtPluginFieldType", "TensorRtPluginRegistrySource" }
        },
        { "TensorRtPluginFieldInfo.cs", new[] { "TensorRtPluginFieldInfo" } },
        { "TensorRtPluginCreatorInfo.cs", new[] { "TensorRtPluginCreatorInfo" } },
        { "TensorRtPluginCreatorSummary.cs", new[] { "TensorRtPluginCreatorSummary" } },
        { "TensorRtPluginFieldSummary.cs", new[] { "TensorRtPluginFieldSummary" } },
        {
            "TensorRtPluginRegistryInventoryDiagnostics.cs",
            new[] { "TensorRtPluginRegistryInventoryDiagnostics" }
        },
        {
            "TensorRtPluginRegistryInventory.cs",
            new[] { "TensorRtPluginRegistryInventory" }
        }
    };

    [Theory]
    [MemberData(nameof(EnvironmentFeatureMethods))]
    public void EnvironmentProbeFeaturePartialsOwnExactStaticMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Diagnostics", $"TensorRtEnvironmentProbe.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicStaticMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(PluginTypeFiles))]
    public void PluginInventoryTypesLiveInTheirDedicatedFiles(
        string fileName,
        string[] expectedTypes)
    {
        string source = ReadSource("Plugins", fileName);
        Assert.Equal(expectedTypes, EnumeratePublicTopLevelTypeNames(source));
    }

    [Fact]
    public void EnvironmentProbeCoreRetainsOnlyCrossFeatureProbeHelpers()
    {
        string core = ReadSource("Diagnostics", "TensorRtEnvironmentProbe.cs");

        Assert.Empty(EnumeratePublicStaticMethodNames(core));
        Assert.Contains("private static bool IsProbeException(", core, StringComparison.Ordinal);
        Assert.Contains("private static string FormatProbeException(", core, StringComparison.Ordinal);
        Assert.Contains("private static bool TryAddProbeStage<T>(", core, StringComparison.Ordinal);
    }

    [Fact]
    public void PluginInventoryMainFileRetainsOnlyTheAggregateType()
    {
        string inventory = ReadSource("Plugins", "TensorRtPluginRegistryInventory.cs");

        Assert.Equal(
            new[] { "TensorRtPluginRegistryInventory" },
            EnumeratePublicTopLevelTypeNames(inventory));
        Assert.Equal(
            new[]
            {
                "FindCreator",
                "TryFindCreator",
                "GetCreatorSummaries",
                "GetFieldSummaries",
                "GetDiagnostics"
            },
            EnumeratePublicInstanceMethodNames(inventory));
        Assert.Contains("public override string ToString()", inventory, StringComparison.Ordinal);
    }

    [Fact]
    public void EnvironmentProbeFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Diagnostics", "TensorRtEnvironmentProbe.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtEnvironmentProbe");
        prefix = prefix.Replace(
            "public static partial class TensorRtEnvironmentProbe",
            "public static class TensorRtEnvironmentProbe",
            StringComparison.Ordinal);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        foreach (string feature in EnvironmentFeatureOrder)
        {
            source.Append(ReadPartialBody(
                "Diagnostics",
                $"TensorRtEnvironmentProbe.{feature}.cs",
                "TensorRtEnvironmentProbe"));
        }

        source.Append(coreBody);
        source.Append('}');
        source.Append('\n');
        Assert.Equal(EnvironmentProbeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void PluginInventoryTypeFilesRecomposeTheOriginalSource()
    {
        string inventory = Normalize(ReadSource("Plugins", "TensorRtPluginRegistryInventory.cs"));
        int typeStart = inventory.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(typeStart >= 0);

        string[] files =
        {
            "TensorRtPluginRegistryTypes.cs",
            "TensorRtPluginFieldInfo.cs",
            "TensorRtPluginCreatorInfo.cs",
            "TensorRtPluginCreatorSummary.cs",
            "TensorRtPluginFieldSummary.cs",
            "TensorRtPluginRegistryInventoryDiagnostics.cs"
        };

        StringBuilder source = new();
        source.Append(inventory[..typeStart]);
        foreach (string file in files)
        {
            source.Append(ReadTopLevelSegment(file, finalSegment: false));
        }

        source.Append(ReadTopLevelSegment("TensorRtPluginRegistryInventory.cs", finalSegment: true));
        Assert.Equal(PluginInventoryOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static (string Prefix, string Body) SplitClass(string source, string typeName)
    {
        int declarationStart = source.IndexOf(
            $"public static partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return (source[..(bodyStart + 1)], body[1..]);
    }

    private static string ReadPartialBody(string module, string fileName, string typeName)
    {
        return SplitClass(Normalize(ReadSource(module, fileName)), typeName).Body;
    }

    private static string ReadTopLevelSegment(string fileName, bool finalSegment)
    {
        string source = Normalize(ReadSource("Plugins", fileName));
        int start = source.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..].TrimEnd('\n') + (finalSegment ? "\n" : "\n\n");
    }

    private static string[] EnumeratePublicStaticMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+static\s+[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicInstanceMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+(?!static\s)[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public\s+(?:sealed\s+class|enum)\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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
