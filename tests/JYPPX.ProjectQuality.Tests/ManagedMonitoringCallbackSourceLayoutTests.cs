using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedMonitoringCallbackSourceLayoutTests
{
    private const string LoggerOriginalNormalizedSha256 =
        "680a3192b73524d792bcf4d65deb5821af6b4b373b551160e5490a4929b3da16";
    private const string ProfilerOriginalNormalizedSha256 =
        "22bc5c1c6eb2f8ef5347ba73b853a64cb5cf3ffc9029972a3093e877af000726";
    private const string OriginalUsingHeader =
        "using System;\n" +
        "using System.Runtime.InteropServices;\n" +
        "using System.Text;\n" +
        "using System.Threading;\n" +
        "using JYPPX.TensorRtSharp.Shared.Interop;\n" +
        "using JYPPX.TensorRtSharp.Internal.Handles;\n" +
        "using JYPPX.TensorRtSharp.Internal.Interop;\n\n" +
        "namespace JYPPX.TensorRtSharp;\n\n";

    public static TheoryData<string, string[]> LoggerFileMethods => new()
    {
        { "TensorRtLogger.cs", Array.Empty<string>() },
        {
            "TensorRtLogger.InterfaceMetadata.cs",
            new[] { "TryGetInterfaceInfo", "TryGetInterfaceInfo", "TryGetApiLanguage", "TryGetApiLanguage" }
        },
        { "TensorRtLogger.Diagnostics.cs", new[] { "EmitDiagnostic" } },
        {
            "TensorRtLogger.Lifecycle.cs",
            new[] { "Dispose", "AttachBorrower", "DetachBorrower", "FreeCallbackState", "ReleaseHandle", "ThrowIfDisposed" }
        },
        {
            "TensorRtLogger.Trampoline.cs",
            new[] { "InvokeManagedLogger", "DecodeUtf8", "RecordInvocation", "RecordFailure" }
        }
    };

    public static TheoryData<string, string[]> ProfilerFileMethods => new()
    {
        { "TensorRtProfiler.cs", Array.Empty<string>() },
        {
            "TensorRtProfiler.InterfaceMetadata.cs",
            new[]
            {
                "TryGetInterfaceInfo", "TryGetInterfaceInfo", "TryGetApiLanguage", "TryGetApiLanguage",
                "GetInterfaceMetadataSnapshot"
            }
        },
        { "TensorRtProfiler.Diagnostics.cs", new[] { "EmitDiagnostic" } },
        {
            "TensorRtProfiler.Lifecycle.cs",
            new[] { "Dispose", "AttachBorrower", "DetachBorrower", "ThrowIfDisposed", "ReleaseHandle", "FreeCallbackState" }
        },
        {
            "TensorRtProfiler.Trampoline.cs",
            new[] { "InvokeManagedProfiler", "DecodeUtf8", "RecordInvocation", "RecordFailure" }
        }
    };

    public static TheoryData<string, string[]> PropertyOwners => new()
    {
        {
            "TensorRtLogger.cs",
            new[]
            {
                "Line", "HasManagedCallback", "CallbackInvocationCount", "CallbackFailureCount",
                "LastCallbackException", "IsAttached"
            }
        },
        { "TensorRtLogger.InterfaceMetadata.cs", new[] { "InterfaceInfo", "ApiLanguage" } },
        {
            "TensorRtLogger.Trampoline.cs",
            new[] { "Handler", "InvocationCount", "FailureCount", "LastException" }
        },
        {
            "TensorRtProfiler.cs",
            new[] { "Line", "IsAttached", "CallbackInvocationCount", "CallbackFailureCount", "LastCallbackException" }
        },
        { "TensorRtProfiler.InterfaceMetadata.cs", new[] { "InterfaceInfo", "ApiLanguage" } },
        {
            "TensorRtProfiler.Trampoline.cs",
            new[] { "Handler", "InvocationCount", "FailureCount", "LastException" }
        }
    };

    [Theory]
    [MemberData(nameof(LoggerFileMethods))]
    public void LoggerPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(ProfilerFileMethods))]
    public void ProfilerPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(PropertyOwners))]
    public void OwnerFilesKeepExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void PublicEnumAndHandlersHaveDedicatedTopLevelOwners()
    {
        string severity = ReadSource("TensorRtLogSeverity.cs");
        string loggerHandler = ReadSource("TensorRtLogHandler.cs");
        string profilerHandler = ReadSource("TensorRtProfilerHandler.cs");

        Assert.Single(Regex.Matches(severity, "^public enum TensorRtLogSeverity$", RegexOptions.Multiline));
        Assert.Equal(
            new[] { "InternalError=0", "Error=1", "Warning=2", "Info=3", "Verbose=4" },
            Regex.Matches(severity, @"^\s*(?<name>[A-Za-z]+)\s*=\s*(?<value>\d+),?$", RegexOptions.Multiline)
                .Select(match => $"{match.Groups["name"].Value}={match.Groups["value"].Value}")
                .ToArray());
        Assert.Single(Regex.Matches(
            loggerHandler,
            @"^public delegate void TensorRtLogHandler\(TensorRtLogSeverity severity, string message\);$",
            RegexOptions.Multiline));
        Assert.Single(Regex.Matches(
            profilerHandler,
            @"^public delegate void TensorRtProfilerHandler\(string layerName, float milliseconds\);$",
            RegexOptions.Multiline));

        foreach (string source in new[] { severity, loggerHandler, profilerHandler })
        {
            Assert.Single(Regex.Matches(source, @"^public (?:enum|delegate void) ", RegexOptions.Multiline));
            Assert.DoesNotContain("public sealed class", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CallbackStateStaysWithItsTrampolineAndHandleStaysWithCore()
    {
        AssertCallbackStateOwner("TensorRtLogger", "TensorRtLoggerCallback", "TensorRtLogHandler");
        AssertCallbackStateOwner("TensorRtProfiler", "TensorRtProfilerCallback", "TensorRtProfilerHandler");
    }

    [Fact]
    public void EvidenceConsumersEnumerateBothCompleteSourceSets()
    {
        string readiness = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimePackageReadiness.ps1"));
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string[] sourceFiles =
        {
            "TensorRtLogger.cs",
            "TensorRtLogger.InterfaceMetadata.cs",
            "TensorRtLogger.Diagnostics.cs",
            "TensorRtLogger.Lifecycle.cs",
            "TensorRtLogger.Trampoline.cs",
            "TensorRtLogSeverity.cs",
            "TensorRtLogHandler.cs",
            "TensorRtProfiler.cs",
            "TensorRtProfiler.InterfaceMetadata.cs",
            "TensorRtProfiler.Diagnostics.cs",
            "TensorRtProfiler.Lifecycle.cs",
            "TensorRtProfiler.Trampoline.cs",
            "TensorRtProfilerHandler.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "ManagedLoggerCallbackBoundaryTests.cs",
                     "ManagedProfilerCallbackBoundaryTests.cs",
                     "DeferredProfilerInterfaceProofClosureTests.cs",
                     "DeferredBTier41To45ProofClosureTests.cs",
                     "VersionedInterfaceApiLanguageReadonlyUpliftTests.cs"
                 })
        {
            string consumer = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "tests",
                "JYPPX.ProjectQuality.Tests",
                consumerFile));
            Assert.Contains("RepositorySourceReader.Read", consumer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void LoggerFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(OriginalUsingHeader);
        source.Append(ReadTopLevelSegment("TensorRtLogSeverity.cs", "/// <summary>"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtLogHandler.cs", "/// <summary>"));
        source.Append('\n');

        string core = Normalize(ReadSource("TensorRtLogger.cs"));
        int ownerStart = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class TensorRtLogger", ownerStart, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(ownerStart >= 0 && declaration > ownerStart && bodyStart > declaration);
        source.Append(core[ownerStart..(bodyStart + 1)].Replace(
            "public sealed partial class TensorRtLogger",
            "public sealed class TensorRtLogger",
            StringComparison.Ordinal));
        source.Append('\n');

        string coreBody = ReadPartialBody("TensorRtLogger.cs", "TensorRtLogger");
        int lineProperty = FindMemberStart(coreBody, "    public TensorRtApiLine Line");
        string[] lifecycle = ReadMemberSegments(
            "TensorRtLogger.Lifecycle.cs",
            "TensorRtLogger",
            "    ~TensorRtLogger()",
            "    public void Dispose(",
            "    internal void AttachBorrower(",
            "    internal void DetachBorrower(",
            "    private void FreeCallbackState(",
            "    private void ReleaseHandle(",
            "    internal void ThrowIfDisposed(");
        string[] trampoline = ReadMemberSegments(
            "TensorRtLogger.Trampoline.cs",
            "TensorRtLogger",
            "    private static BridgeStatusCode InvokeManagedLogger(",
            "    private static string DecodeUtf8(",
            "    private sealed class CallbackState");

        source.Append(coreBody[..lineProperty]);
        source.Append(lifecycle[0]);
        source.Append(coreBody[lineProperty..]);
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtLogger.InterfaceMetadata.cs", "TensorRtLogger"));
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtLogger.Diagnostics.cs", "TensorRtLogger"));
        source.Append('\n');
        source.Append(lifecycle[1]);
        source.Append(lifecycle[2]);
        source.Append(lifecycle[3]);
        source.Append(trampoline[0]);
        source.Append(trampoline[1]);
        source.Append(lifecycle[4]);
        source.Append(lifecycle[5]);
        source.Append(lifecycle[6]);
        source.Append('\n');
        source.Append(trampoline[2]);
        source.Append("}\n");

        Assert.Equal(LoggerOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void ProfilerFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(OriginalUsingHeader);
        source.Append(ReadTopLevelSegment("TensorRtProfilerHandler.cs", "/// <summary>"));
        source.Append('\n');

        string core = Normalize(ReadSource("TensorRtProfiler.cs"));
        int ownerStart = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class TensorRtProfiler", ownerStart, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(ownerStart >= 0 && declaration > ownerStart && bodyStart > declaration);
        source.Append(core[ownerStart..(bodyStart + 1)].Replace(
            "public sealed partial class TensorRtProfiler",
            "public sealed class TensorRtProfiler",
            StringComparison.Ordinal));
        source.Append('\n');

        string coreBody = ReadPartialBody("TensorRtProfiler.cs", "TensorRtProfiler");
        int lineProperty = FindMemberStart(coreBody, "    public TensorRtApiLine Line");
        string[] lifecycle = ReadMemberSegments(
            "TensorRtProfiler.Lifecycle.cs",
            "TensorRtProfiler",
            "    ~TensorRtProfiler()",
            "    public void Dispose(",
            "    internal void AttachBorrower(",
            "    internal void DetachBorrower(",
            "    internal void ThrowIfDisposed(",
            "    private void ReleaseHandle(",
            "    private void FreeCallbackState(");
        string[] trampoline = ReadMemberSegments(
            "TensorRtProfiler.Trampoline.cs",
            "TensorRtProfiler",
            "    private static BridgeStatusCode InvokeManagedProfiler(",
            "    private static string DecodeUtf8(",
            "    private sealed class CallbackState");

        source.Append(coreBody[..lineProperty]);
        source.Append(lifecycle[0]);
        source.Append(coreBody[lineProperty..]);
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtProfiler.InterfaceMetadata.cs", "TensorRtProfiler"));
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtProfiler.Diagnostics.cs", "TensorRtProfiler"));
        source.Append('\n');
        source.Append(lifecycle[1]);
        source.Append(lifecycle[2]);
        source.Append(lifecycle[3]);
        source.Append(lifecycle[4]);
        source.Append(trampoline[0]);
        source.Append(trampoline[1]);
        source.Append(lifecycle[5]);
        source.Append(lifecycle[6]);
        source.Append('\n');
        source.Append(trampoline[2]);
        source.Append("}\n");

        Assert.Equal(ProfilerOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static void AssertCallbackStateOwner(string typeName, string nativeCallbackType, string handlerType)
    {
        string trampoline = ReadSource($"{typeName}.Trampoline.cs");
        string core = ReadSource($"{typeName}.cs");
        Assert.Single(Regex.Matches(trampoline, "^    private sealed class CallbackState$", RegexOptions.Multiline));
        Assert.Contains($"public CallbackState({handlerType} handler)", trampoline, StringComparison.Ordinal);
        Assert.Single(Regex.Matches(
            core,
            $@"^    private readonly {Regex.Escape(nativeCallbackType)}\?? _nativeCallback;$",
            RegexOptions.Multiline));
        Assert.Contains("internal SafeTensorRtObjectHandle Handle => _handle;", core, StringComparison.Ordinal);

        foreach (string suffix in new[] { string.Empty, ".InterfaceMetadata", ".Diagnostics", ".Lifecycle" })
        {
            string source = ReadSource($"{typeName}{suffix}.cs");
            Assert.DoesNotContain("private sealed class CallbackState", source, StringComparison.Ordinal);
        }
    }

    private static string[] ReadMemberSegments(string fileName, string typeName, params string[] declarationMarkers)
    {
        string body = ReadPartialBody(fileName, typeName);
        int[] starts = declarationMarkers.Select(marker => FindMemberStart(body, marker)).ToArray();
        Assert.Equal(starts.OrderBy(value => value), starts);
        return starts.Select((start, index) =>
        {
            int end = index + 1 < starts.Length ? starts[index + 1] : body.Length;
            return body[start..end];
        }).ToArray();
    }

    private static int FindMemberStart(string body, string declarationMarker)
    {
        int declaration = body.IndexOf(declarationMarker, StringComparison.Ordinal);
        Assert.True(declaration >= 0, $"Member marker not found: {declarationMarker}");
        int summary = body.LastIndexOf("    /// <summary>", declaration, StringComparison.Ordinal);
        int previousMemberClose = body.LastIndexOf("\n    }", declaration, StringComparison.Ordinal);
        return summary > previousMemberClose ? summary : declaration;
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
            "JYPPX.TensorRtSharp",
            "Callbacks",
            "Monitoring",
            fileName));
    }
}
