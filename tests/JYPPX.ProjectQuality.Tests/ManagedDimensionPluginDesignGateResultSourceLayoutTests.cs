using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDimensionPluginDesignGateResultSourceLayoutTests
{
    private const string DimensionOriginalNormalizedSha256 =
        "0b998614fc741b7a016dd273a3ab061fe849eb574b06375b6c78268db8069f80";
    private const string PluginOriginalNormalizedSha256 =
        "23f0571f784ffd698cedf3ee3613753e59c82570dd5ab65ebe2140785dd5ce59";

    public static TheoryData<string, string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Network",
            "TensorRtDimensionExpressionSnapshotDesignGate.cs",
            "TensorRtDimensionExpressionSnapshotDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "Network",
            "TensorRtDimensionExpressionSnapshotDesignGateResult.cs",
            "TensorRtDimensionExpressionSnapshotDesignGateResult",
            new[] { "ToString" }
        },
        {
            "Plugins",
            "TensorRtPluginCreatorV3MetadataDesignGate.cs",
            "TensorRtPluginCreatorV3MetadataDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "Plugins",
            "TensorRtPluginCreatorV3MetadataDesignGateResult.cs",
            "TensorRtPluginCreatorV3MetadataDesignGateResult",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> ResultProperties => new()
    {
        {
            "Network",
            "TensorRtDimensionExpressionSnapshotDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsDimensionExpression", "LineSupportsSizeTensor",
                "SnapshotTypeReady", "ConstantSnapshotCopyReady", "SizeTensorMetadataCopyReady",
                "OwnerLifetimeKnown", "ExprBuilderOwnershipModeled", "PluginShapeCallbackLifetimeModeled",
                "ExpressionPointerExposed", "ExpressionPointerProduced", "BorrowedExpressionPointerEscaped",
                "ExprBuilderPointerExposed", "ExprBuilderCreationEnabled", "ExpressionNodePublicOwnershipControl",
                "DirectDimensionExpressionRowsDeferred", "DirectExpressionBuilderRowsDeferred",
                "PointerFreeSurfaceReady", "CopiedMetadataShapeReady", "DesignGateReady",
                "CanPromoteWithoutDesignGate", "CanPromoteWithoutRuntimeProof",
                "FullPackageConsumerRuntimeEvidenceReady", "CanPromoteRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "CandidateInterfaces", "CandidateMethods", "RequiredOutputMode",
                "NextSafeImplementationStep", "CandidateMethodCount", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "Plugins",
            "TensorRtPluginCreatorV3MetadataDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsPluginCreatorV3", "RegistryInventorySnapshotReady",
                "CopiedIdentityReady", "CopiedFieldMetadataReady", "CopiedInterfaceInfoReady",
                "PluginCreationModeled", "BorrowedCreatorLifetimeModeled", "PluginCreatorPointerExposed",
                "BorrowedPluginCreatorHandleExposed", "PluginInstanceCreationEnabled",
                "PluginResourceOwnershipControlEnabled", "DirectCreatePluginRowsDeferred",
                "DirectBorrowedCreatorListRowsDeferred", "CopiedMetadataShapeReady", "PointerFreeSurfaceReady",
                "DesignGateReady", "CanPromoteWithoutRuntimeProof", "FullPackageConsumerRuntimeEvidenceReady",
                "CanPromoteRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired",
                "CanDeleteDeferredRecord", "CandidateInterfaces", "CandidateMethods", "RequiredOutputMode",
                "NextSafeImplementationStep", "CandidateMethodCount", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        }
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
    [MemberData(nameof(ResultProperties))]
    public void ResultsKeepExactPublicProperties(string module, string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(module, fileName)));
    }

    [Fact]
    public void ResultsKeepOneInternalConstructorEach()
    {
        foreach ((string module, string fileName, string typeName) in new[]
                 {
                     ("Network", "TensorRtDimensionExpressionSnapshotDesignGateResult.cs",
                         "TensorRtDimensionExpressionSnapshotDesignGateResult"),
                     ("Plugins", "TensorRtPluginCreatorV3MetadataDesignGateResult.cs",
                         "TensorRtPluginCreatorV3MetadataDesignGateResult")
                 })
        {
            Assert.Single(Regex.Matches(
                ReadSource(module, fileName),
                $@"^    internal {typeName}\(",
                RegexOptions.Multiline));
        }
    }

    [Fact]
    public void EvaluatorsDoNotOwnResultDeclarations()
    {
        Assert.DoesNotContain(
            "public readonly struct TensorRtDimensionExpressionSnapshotDesignGateResult",
            ReadSource("Network", "TensorRtDimensionExpressionSnapshotDesignGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtPluginCreatorV3MetadataDesignGateResult",
            ReadSource("Plugins", "TensorRtPluginCreatorV3MetadataDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFreeAndNonProof()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Network", "TensorRtDimensionExpressionSnapshotDesignGateResult.cs"),
                     ("Plugins", "TensorRtPluginCreatorV3MetadataDesignGateResult.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
            Assert.Contains("IsRuntimeExecutionProof => false", source, StringComparison.Ordinal);
            Assert.Contains("CanPromoteRuntimeProof => false", source, StringComparison.Ordinal);
            Assert.Contains("DeferredRowsStillRequired => true", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderReadinessCandidatesAndConsumersUseCompleteSourceSets()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string readiness = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-RuntimePackageReadiness.ps1"));
        string candidates = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-readonly-candidate-list.json"));

        foreach (string sourceFile in new[]
                 {
                     "TensorRtDimensionExpressionSnapshotDesignGate.cs",
                     "TensorRtDimensionExpressionSnapshotDesignGateResult.cs",
                     "TensorRtPluginCreatorV3MetadataDesignGate.cs",
                     "TensorRtPluginCreatorV3MetadataDesignGateResult.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        Assert.Contains("TensorRtDimensionExpressionSnapshotDesignGateResult.cs", readiness, StringComparison.Ordinal);
        Assert.Contains("TensorRtDimensionExpressionSnapshotDesignGateResult.cs", candidates, StringComparison.Ordinal);
        Assert.Contains("TensorRtPluginCreatorV3MetadataDesignGateResult.cs", candidates, StringComparison.Ordinal);

        foreach (string consumerName in new[]
                 {
                     "DimensionExpressionSnapshotDesignGateTests.cs",
                     "DeferredReadonlyUpliftBatchTests.cs"
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
    public void SpecializedDocsAndSourceOrganizationReferenceDedicatedResults()
    {
        string dimensionDocs = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "dimension-expression-snapshot-design-gate.md"));
        string pluginDocs = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "plugin-creator-v3-metadata-design-gate.md"));
        string sourceOrganization = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "source-organization.md"));

        Assert.Contains("TensorRtDimensionExpressionSnapshotDesignGateResult", dimensionDocs, StringComparison.Ordinal);
        Assert.Contains("TensorRtPluginCreatorV3MetadataDesignGateResult.cs", pluginDocs, StringComparison.Ordinal);
        Assert.Contains("*DesignGateResult", sourceOrganization, StringComparison.Ordinal);
    }

    [Fact]
    public void DimensionFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Network",
            "TensorRtDimensionExpressionSnapshotDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Network",
            "TensorRtDimensionExpressionSnapshotDesignGateResult.cs"));
        Assert.Equal(DimensionOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void PluginFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Plugins",
            "TensorRtPluginCreatorV3MetadataDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(
            "Plugins",
            "TensorRtPluginCreatorV3MetadataDesignGateResult.cs"));
        Assert.Equal(PluginOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] EnumerateTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                Normalize(source),
                @"^public (?:static class|readonly struct) (?<name>[A-Za-z_][A-Za-z0-9_]*)$",
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

    private static string ReadTopLevelSegment(string module, string fileName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int start = source.IndexOf("/// <summary>", StringComparison.Ordinal);
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
