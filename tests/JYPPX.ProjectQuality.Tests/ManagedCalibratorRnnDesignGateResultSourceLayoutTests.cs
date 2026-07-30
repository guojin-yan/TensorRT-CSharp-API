using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCalibratorRnnDesignGateResultSourceLayoutTests
{
    private const string CalibratorOriginalNormalizedSha256 =
        "d9f624227e9d1486ad4a2b6bf42ec979a517637652fa418ae8cec2a94b1af5f4";
    private const string RnnOriginalNormalizedSha256 =
        "25b435da28b6a6476302556065f0bf253dcd68e5b0f6b967fa71696fed7e0e0c";

    public static TheoryData<string, string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Builder",
            "TensorRtCalibratorMetadataDesignGate.cs",
            "TensorRtCalibratorMetadataDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "Builder",
            "TensorRtCalibratorMetadataDesignGateResult.cs",
            "TensorRtCalibratorMetadataDesignGateResult",
            new[] { "ToString" }
        },
        {
            "Layers",
            "TensorRtRnnV2BorrowedStateDesignGate.cs",
            "TensorRtRnnV2BorrowedStateDesignGate",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        {
            "Layers",
            "TensorRtRnnV2BorrowedStateDesignGateResult.cs",
            "TensorRtRnnV2BorrowedStateDesignGateResult",
            Array.Empty<string>()
        }
    };

    public static TheoryData<string, string, string[]> ResultProperties => new()
    {
        {
            "Builder",
            "TensorRtCalibratorMetadataDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsCalibrator", "PresenceProbeAvailable",
                "CopiedAlgorithmMetadataReady", "CopiedInterfaceInfoMetadataReady", "BatchCallbackOwnershipModeled",
                "CacheBufferOwnershipModeled", "CalibratorPointerExposed", "CalibratorPointerProduced",
                "BorrowedCalibratorPointerEscaped", "CallbackInvocationEnabled", "BatchBufferAccessEnabled",
                "CacheBufferAccessEnabled", "DirectCalibratorCallbackRowsDeferred", "DirectCalibratorCacheRowsDeferred",
                "PointerFreeSurfaceReady", "CopiedMetadataShapeReady", "DesignGateReady",
                "CanPromoteWithoutDesignGate", "CanPromoteWithoutRuntimeProof",
                "FullPackageConsumerRuntimeEvidenceReady", "CanPromoteRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "Layers",
            "TensorRtRnnV2BorrowedStateDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "Line", "LineSupportsRnnV2",
                "DataLengthScalarPromoted", "NetworkOwnedTensorReferencePolicyReady", "GateWeightSnapshotCopyReady",
                "OwnerLifetimeKnown", "BorrowedTensorPointerExposed", "BorrowedWeightsPointerExposed",
                "BorrowedStateEscapesCall", "PointerFreeSurfaceReady", "BorrowedSnapshotPromotionReady",
                "DesignGateReady", "IsRuntimeExecutionEvidence", "IsRuntimeExecutionProof", "CanPromoteRuntimeProof",
                "DeferredBorrowedRowsStillRequired", "SelectedTriageRowCount", "PromotedScalarTriageRowCount",
                "RemainingDeferredTriageRowCount", "SelectedCandidateMethods", "DeferredBorrowedMethods",
                "BlockedPrerequisites", "Status", "NextSafeImplementationStep", "Diagnostic"
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
                     ("Builder", "TensorRtCalibratorMetadataDesignGateResult.cs",
                         "TensorRtCalibratorMetadataDesignGateResult"),
                     ("Layers", "TensorRtRnnV2BorrowedStateDesignGateResult.cs",
                         "TensorRtRnnV2BorrowedStateDesignGateResult")
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
            "public readonly struct TensorRtCalibratorMetadataDesignGateResult",
            ReadSource("Builder", "TensorRtCalibratorMetadataDesignGate.cs"),
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "public readonly struct TensorRtRnnV2BorrowedStateDesignGateResult",
            ReadSource("Layers", "TensorRtRnnV2BorrowedStateDesignGate.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void DedicatedResultsRemainPointerFreeAndNonProof()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Builder", "TensorRtCalibratorMetadataDesignGateResult.cs"),
                     ("Layers", "TensorRtRnnV2BorrowedStateDesignGateResult.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
            Assert.Contains("IsRuntimeExecutionProof => false", source, StringComparison.Ordinal);
            Assert.Contains("CanPromoteRuntimeProof => false", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderReadinessCandidateAndConsumersUseCompleteSourceSets()
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
                     "TensorRtCalibratorMetadataDesignGate.cs",
                     "TensorRtCalibratorMetadataDesignGateResult.cs",
                     "TensorRtRnnV2BorrowedStateDesignGate.cs",
                     "TensorRtRnnV2BorrowedStateDesignGateResult.cs"
                 })
        {
            Assert.Contains(sourceFile, reader, StringComparison.Ordinal);
        }

        Assert.Contains("TensorRtCalibratorMetadataDesignGateResult.cs", readiness, StringComparison.Ordinal);
        Assert.Contains("TensorRtCalibratorMetadataDesignGateResult.cs", candidates, StringComparison.Ordinal);

        foreach (string consumerName in new[]
                 {
                     "CalibratorMetadataDesignGateTests.cs",
                     "RnnV2BorrowedStateDesignGateTests.cs"
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
        string calibratorDocs = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "calibrator-metadata-design-gate.md"));
        string rnnDocs = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "rnnv2-borrowed-state-design-gate.md"));
        string sourceOrganization = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "source-organization.md"));

        Assert.Contains("TensorRtCalibratorMetadataDesignGateResult.cs", calibratorDocs, StringComparison.Ordinal);
        Assert.Contains("TensorRtRnnV2BorrowedStateDesignGateResult.cs", rnnDocs, StringComparison.Ordinal);
        Assert.Contains("Calibrator metadata", sourceOrganization, StringComparison.Ordinal);
        Assert.Contains("RNNv2 borrowed-state", sourceOrganization, StringComparison.Ordinal);
    }

    [Fact]
    public void CalibratorFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Builder",
            "TensorRtCalibratorMetadataDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Builder", "TensorRtCalibratorMetadataDesignGateResult.cs"));
        Assert.Equal(CalibratorOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void RnnFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource(
            "Layers",
            "TensorRtRnnV2BorrowedStateDesignGate.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Layers", "TensorRtRnnV2BorrowedStateDesignGateResult.cs"));
        Assert.Equal(RnnOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
