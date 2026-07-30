using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedDependencyRuntimeProbeSourceLayoutTests
{
    private const string DependencyOriginalNormalizedSha256 =
        "213076b3d11b6e80e467e0f731f25a5d653dbe92355c4d7cc6dfcb6ccdd73a4b";
    private const string RuntimeOriginalNormalizedSha256 =
        "37cc1851f0700b5ac2048c84bc03d6d40bf272f1be83a2aff8abb63fd07d34c8";

    public static TheoryData<string, string, string, string[]> FileOwnersAndMethods => new()
    {
        {
            "Diagnostics",
            "TensorRtNativeDependencySource.cs",
            "TensorRtNativeDependencySource",
            Array.Empty<string>()
        },
        {
            "Diagnostics",
            "TensorRtNativeDependencyInfo.cs",
            "TensorRtNativeDependencyInfo",
            new[] { "ToString" }
        },
        {
            "Diagnostics",
            "TensorRtDependencyProbeReport.cs",
            "TensorRtDependencyProbeReport",
            new[] { "ToString" }
        },
        {
            "Runtime",
            "TensorRtGlobalRuntimeVersion.cs",
            "TensorRtGlobalRuntimeVersion",
            new[] { "ToString" }
        },
        {
            "Runtime",
            "TensorRtRuntimeProbeStage.cs",
            "TensorRtRuntimeProbeStage",
            new[] { "ToString" }
        },
        {
            "Runtime",
            "TensorRtRuntimeProbeReport.cs",
            "TensorRtRuntimeProbeReport",
            new[] { "ToString" }
        }
    };

    public static TheoryData<string, string, string[]> PublicProperties => new()
    {
        {
            "Diagnostics",
            "TensorRtNativeDependencyInfo.cs",
            new[] { "Source", "Name", "Path", "Exists", "FileVersion", "ProductVersion", "Diagnostic" }
        },
        {
            "Diagnostics",
            "TensorRtDependencyProbeReport.cs",
            new[]
            {
                "Line", "BridgeInitialized", "BridgeDiagnostic", "NativeBridgeCandidates", "LoadedModules",
                "SearchPathCandidates", "Diagnostics", "LoadedModuleCount", "SearchPathCandidateCount"
            }
        },
        {
            "Runtime",
            "TensorRtGlobalRuntimeVersion.cs",
            new[]
            {
                "Line", "InferLibVersion", "Major", "Minor", "Patch", "Build", "OnnxParserVersion",
                "HasGlobalLogger"
            }
        },
        {
            "Runtime",
            "TensorRtRuntimeProbeStage.cs",
            new[] { "Name", "Succeeded", "Message" }
        },
        {
            "Runtime",
            "TensorRtRuntimeProbeReport.cs",
            new[]
            {
                "Line", "GlobalVersion", "GlobalPluginRegistry", "Stages", "RuntimeCreationSucceeded",
                "FirstFailure"
            }
        }
    };

    public static TheoryData<string, string, string> ConstructorOwners => new()
    {
        { "Diagnostics", "TensorRtNativeDependencyInfo.cs", "TensorRtNativeDependencyInfo" },
        { "Diagnostics", "TensorRtDependencyProbeReport.cs", "TensorRtDependencyProbeReport" },
        { "Runtime", "TensorRtGlobalRuntimeVersion.cs", "TensorRtGlobalRuntimeVersion" },
        { "Runtime", "TensorRtRuntimeProbeStage.cs", "TensorRtRuntimeProbeStage" },
        { "Runtime", "TensorRtRuntimeProbeReport.cs", "TensorRtRuntimeProbeReport" }
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
    public void FilesKeepExactPublicPropertyOrder(string module, string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(module, fileName)));
    }

    [Theory]
    [MemberData(nameof(ConstructorOwners))]
    public void ClassesKeepOneInternalConstructorWithTheirNamesakeTypes(
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
    public void ReportFilesDoNotRetainEarlierTypeDeclarations()
    {
        string dependency = ReadSource("Diagnostics", "TensorRtDependencyProbeReport.cs");
        Assert.DoesNotContain("public enum TensorRtNativeDependencySource", dependency, StringComparison.Ordinal);
        Assert.DoesNotContain("public sealed class TensorRtNativeDependencyInfo", dependency, StringComparison.Ordinal);

        string runtime = ReadSource("Runtime", "TensorRtRuntimeProbeReport.cs");
        Assert.DoesNotContain("public sealed class TensorRtGlobalRuntimeVersion", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("public sealed class TensorRtRuntimeProbeStage", runtime, StringComparison.Ordinal);
    }

    [Fact]
    public void ProbeModelsRemainPointerFreeCopiedDiagnostics()
    {
        foreach ((string module, string fileName) in new[]
                 {
                     ("Diagnostics", "TensorRtNativeDependencySource.cs"),
                     ("Diagnostics", "TensorRtNativeDependencyInfo.cs"),
                     ("Diagnostics", "TensorRtDependencyProbeReport.cs"),
                     ("Runtime", "TensorRtGlobalRuntimeVersion.cs"),
                     ("Runtime", "TensorRtRuntimeProbeStage.cs"),
                     ("Runtime", "TensorRtRuntimeProbeReport.cs")
                 })
        {
            string source = ReadSource(module, fileName);
            Assert.DoesNotContain("public IntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public UIntPtr", source, StringComparison.Ordinal);
            Assert.DoesNotContain("public nint", source, StringComparison.Ordinal);
        }

        Assert.Contains(
            "Read-only TensorRT global runtime version details",
            ReadSource("Runtime", "TensorRtGlobalRuntimeVersion.cs"),
            StringComparison.Ordinal);
        Assert.Contains(
            "Non-throwing diagnostic report",
            ReadSource("Diagnostics", "TensorRtDependencyProbeReport.cs"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void ReaderKeepsBothThreeFileSourceSetsInHistoricalDeclarationOrder()
    {
        string reader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));

        AssertInOrder(
            reader,
            "TensorRtNativeDependencySource.cs",
            "TensorRtNativeDependencyInfo.cs",
            "TensorRtDependencyProbeReport.cs");
        AssertInOrder(
            reader,
            "TensorRtGlobalRuntimeVersion.cs",
            "TensorRtRuntimeProbeStage.cs",
            "TensorRtRuntimeProbeReport.cs");
    }

    [Fact]
    public void ProducerConsumerAndCandidateEvidenceReferenceCompleteTypeSets()
    {
        string producer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Diagnostics",
            "TensorRtEnvironmentProbe.DependencyProbes.cs"));
        foreach (string constructor in new[]
                 {
                     "new TensorRtNativeDependencyInfo(",
                     "new TensorRtDependencyProbeReport(",
                     "new TensorRtGlobalRuntimeVersion(",
                     "new TensorRtRuntimeProbeStage(",
                     "new TensorRtRuntimeProbeReport("
                 })
        {
            Assert.Contains(constructor, producer, StringComparison.Ordinal);
        }

        string consumer = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "ReadonlyDiagnosticsCandidateImplementationEvidenceTests.cs"));
        Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);

        string candidate = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-readonly-candidate-list.json"));
        AssertInOrder(
            candidate,
            "TensorRtNativeDependencySource.cs",
            "TensorRtNativeDependencyInfo.cs",
            "TensorRtDependencyProbeReport.cs");
        Assert.Contains("\"isRuntimeProof\": false", candidate, StringComparison.Ordinal);
    }

    [Fact]
    public void DocumentationReferencesDedicatedProbeTypeFilesAndNonProofBoundary()
    {
        foreach (string organizationPath in new[]
                 {
                     Path.Combine("docs", "articles", "en", "source-organization.md"),
                     Path.Combine("docs", "articles", "zh-cn", "source-organization.md")
                 })
        {
            string organization = File.ReadAllText(Path.Combine(RepositoryPaths.Root, organizationPath));
            Assert.Contains("TensorRtNativeDependencySource.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtNativeDependencyInfo.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtGlobalRuntimeVersion.cs", organization, StringComparison.Ordinal);
            Assert.Contains("TensorRtRuntimeProbeStage.cs", organization, StringComparison.Ordinal);
        }

        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "windows-api-completion.md"));
        Assert.Contains("TensorRtDependencyProbeReport.cs", article, StringComparison.Ordinal);
        Assert.Contains("TensorRtRuntimeProbeReport.cs", article, StringComparison.Ordinal);
        Assert.Contains("不构成 runtime 或 release proof", article, StringComparison.Ordinal);
    }

    [Fact]
    public void DependencyFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Diagnostics", "TensorRtNativeDependencySource.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Diagnostics", "TensorRtNativeDependencyInfo.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Diagnostics", "TensorRtDependencyProbeReport.cs"));
        Assert.Equal(DependencyOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void RuntimeFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(Normalize(ReadSource("Runtime", "TensorRtGlobalRuntimeVersion.cs")));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Runtime", "TensorRtRuntimeProbeStage.cs"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Runtime", "TensorRtRuntimeProbeReport.cs"));
        Assert.Equal(RuntimeOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
