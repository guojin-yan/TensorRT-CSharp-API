using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedMonitoringProgressDesignGateSourceLayoutTests
{
    private const string ProgressMonitorOriginalNormalizedSha256 =
        "615988b4c7c65a15d2ee74815a44d5c0d0d59d58d9f5036e9694ffee70ebe73c";
    private const string ErrorRecorderGateOriginalNormalizedSha256 =
        "5fe4fcc308b043ae5d16c50c03544de40d6e03152203a0858bc66aed59927d3e";
    private const string StreamIoGateOriginalNormalizedSha256 =
        "ce954216836edd26581e33434f8597fb2696fe57a32964cfe171125bca824714";
    private const string ProgressMonitorOriginalUsingHeader =
        "using System;\n" +
        "using System.Runtime.InteropServices;\n" +
        "using System.Text;\n" +
        "using System.Threading;\n" +
        "using JYPPX.TensorRtSharp.Shared.Interop;\n" +
        "using JYPPX.TensorRtSharp.Internal.Handles;\n" +
        "using JYPPX.TensorRtSharp.Internal.Interop;\n\n" +
        "namespace JYPPX.TensorRtSharp;\n\n";

    public static TheoryData<string, string[]> ProgressMonitorFileMethods => new()
    {
        { "TensorRtProgressMonitor.cs", Array.Empty<string>() },
        {
            "TensorRtProgressMonitor.InterfaceMetadata.cs",
            new[] { "TryGetInterfaceInfo", "TryGetInterfaceInfo", "TryGetApiLanguage", "TryGetApiLanguage" }
        },
        { "TensorRtProgressMonitor.Diagnostics.cs", new[] { "EmitDiagnostic" } },
        {
            "TensorRtProgressMonitor.Lifecycle.cs",
            new[] { "Dispose", "AttachBorrower", "DetachBorrower", "ThrowIfDisposed", "ReleaseHandle", "FreeCallbackState" }
        },
        {
            "TensorRtProgressMonitor.Trampoline.cs",
            new[] { "InvokeManagedProgressMonitor", "DecodeUtf8", "DecodeUtf8Nullable", "RecordInvocation", "RecordFailure" }
        }
    };

    public static TheoryData<string, string[]> ProgressMonitorPropertyOwners => new()
    {
        {
            "TensorRtProgressMonitor.cs",
            new[] { "Line", "IsAttached", "CallbackInvocationCount", "CallbackFailureCount", "LastCallbackException" }
        },
        { "TensorRtProgressMonitor.InterfaceMetadata.cs", new[] { "InterfaceInfo", "ApiLanguage" } },
        {
            "TensorRtProgressMonitorEvent.cs",
            new[] { "Kind", "PhaseName", "ParentPhase", "Step", "StepCount" }
        },
        { "TensorRtProgressMonitorDiagnosticResult.cs", new[] { "ShouldContinue", "CallbackAccepted" } },
        {
            "TensorRtProgressMonitor.Trampoline.cs",
            new[] { "Handler", "InvocationCount", "FailureCount", "LastException" }
        }
    };

    public static TheoryData<string, string[]> GateFileMethods => new()
    {
        {
            "TensorRtErrorRecorderDiagnosticsDesignGate.cs",
            new[] { "EvaluateKnownSurface", "Evaluate", "Evaluate", "EvaluateCore" }
        },
        { "TensorRtErrorRecorderDiagnosticsDesignGateResult.cs", new[] { "ToString" } },
        {
            "TensorRtStreamIoInterfaceInfoDesignGate.cs",
            new[] { "EvaluateKnownSurface", "Evaluate" }
        },
        { "TensorRtStreamIoInterfaceInfoDesignGateResult.cs", new[] { "ToString" } }
    };

    public static TheoryData<string, string[]> GateResultProperties => new()
    {
        {
            "TensorRtErrorRecorderDiagnosticsDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsErrorRecorder", "SnapshotTypeAvailable",
                "RuntimeSnapshotAvailable", "RefitterSnapshotAvailable", "PresenceControlsAvailable",
                "ClearControlsAvailable", "CopiedSnapshotObserved", "HasRecorder", "ErrorCount", "HasOverflowed",
                "CopiedRecordCount", "CopiedDiagnosticsReady", "SnapshotRecordCopyReady", "RecorderPointerExposed",
                "RecorderPointerProduced", "BorrowedRecorderPointerEscaped", "RefCountPublicOwnershipControl",
                "InterfaceInfoPublicOwnershipControl", "DirectRecorderOwnershipDeferred", "PointerFreeSurfaceReady",
                "DesignGateReady", "CanPromoteWithoutDesignGate", "CanPromoteWithoutRuntimeProof",
                "FullPackageConsumerRuntimeEvidenceReady", "CanPromoteRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "CandidateInterfaces", "CandidateMethods", "RequiredOutputMode",
                "NextSafeImplementationStep", "CandidateMethodCount", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        },
        {
            "TensorRtStreamIoInterfaceInfoDesignGateResult.cs",
            new[]
            {
                "EvidenceKind", "DiagnosticsKind", "RuntimeEvidenceKind", "IsRuntimeExecutionEvidence",
                "IsRuntimeExecutionProof", "Line", "LineSupportsStreamIo", "LineSupportsStreamWriter",
                "CopiedInterfaceInfoMetadataReady", "StreamOwnerLifetimeModeled", "ReadWriteBufferOwnershipModeled",
                "SeekTellLifetimeModeled", "StreamReaderPointerExposed", "StreamWriterPointerExposed",
                "StreamCallbackInvocationEnabled", "StreamBufferExposed", "DirectStreamCallbackRowsDeferred",
                "ManagedOwnedStreamOwnerLedgerReady", "ManagedOwnerLifetimeReady", "NativeOwnerCreateDestroySymmetric",
                "NoThrowVTableReady", "ExceptionToStatusMappingReady", "DetachBeforeReleaseReady",
                "CanImplementStreamMetadataNow", "StreamCallbackBridgeReady", "CopiedMetadataShapeReady",
                "PointerFreeSurfaceReady", "DesignGateReady", "CanPromoteWithoutRuntimeProof",
                "CanPromoteRuntimeProof", "RuntimeProofBlocked", "DeferredRowsStillRequired", "CandidateInterfaces",
                "CandidateMethods", "DirectCallbackMethods", "ApiLanguageCandidateMethods", "OwnerHandleCandidates",
                "OwnerLifecycleRequirements", "CallbackSafetyRequirements", "OwnerLedgerTrackedMethods",
                "RequiredOutputMode", "NextSafeImplementationStep", "CandidateMethodCount", "DirectCallbackMethodCount",
                "ApiLanguageCandidateMethodCount", "OwnerLedgerTrackedMethodCount", "BlockedPrerequisites",
                "BlockedPrerequisiteCount", "Status", "Diagnostic"
            }
        }
    };

    [Theory]
    [MemberData(nameof(ProgressMonitorFileMethods))]
    public void ProgressMonitorPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(ProgressMonitorPropertyOwners))]
    public void ProgressMonitorFilesOwnExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(GateFileMethods))]
    public void GateFilesOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(GateResultProperties))]
    public void GateResultFilesOwnExactPublicProperties(string fileName, string[] expectedProperties)
    {
        Assert.Single(Regex.Matches(
            ReadSource(fileName),
            @"^public readonly struct TensorRt(?:ErrorRecorderDiagnostics|StreamIoInterfaceInfo)DesignGateResult$",
            RegexOptions.Multiline));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(ReadSource(fileName)));
    }

    [Fact]
    public void ProgressMonitorModelsAndHandlerHaveDedicatedTopLevelOwners()
    {
        string eventKind = ReadSource("TensorRtProgressMonitorEventKind.cs");
        string progressEvent = ReadSource("TensorRtProgressMonitorEvent.cs");
        string diagnosticResult = ReadSource("TensorRtProgressMonitorDiagnosticResult.cs");
        string handler = ReadSource("TensorRtProgressMonitorHandler.cs");

        Assert.Single(Regex.Matches(eventKind, "^public enum TensorRtProgressMonitorEventKind$", RegexOptions.Multiline));
        Assert.Equal(
            new[] { "Unknown=0", "PhaseStart=1", "StepComplete=2", "PhaseFinish=3" },
            Regex.Matches(eventKind, @"^\s*(?<name>[A-Za-z]+)\s*=\s*(?<value>\d+),?$", RegexOptions.Multiline)
                .Select(match => $"{match.Groups["name"].Value}={match.Groups["value"].Value}")
                .ToArray());
        Assert.Single(Regex.Matches(progressEvent, "^public readonly struct TensorRtProgressMonitorEvent$", RegexOptions.Multiline));
        Assert.Single(Regex.Matches(diagnosticResult, "^public readonly struct TensorRtProgressMonitorDiagnosticResult$", RegexOptions.Multiline));
        Assert.Single(Regex.Matches(
            handler,
            @"^public delegate bool TensorRtProgressMonitorHandler\(TensorRtProgressMonitorEvent progressEvent\);$",
            RegexOptions.Multiline));
    }

    [Fact]
    public void ProgressMonitorCallbackStateStaysWithTrampolineAndHandleStaysWithCore()
    {
        string trampoline = ReadSource("TensorRtProgressMonitor.Trampoline.cs");
        string core = ReadSource("TensorRtProgressMonitor.cs");
        Assert.Single(Regex.Matches(trampoline, "^    private sealed class CallbackState$", RegexOptions.Multiline));
        Assert.Contains("public CallbackState(TensorRtProgressMonitorHandler handler)", trampoline, StringComparison.Ordinal);
        Assert.Contains("private readonly TensorRtProgressMonitorCallback _nativeCallback;", core, StringComparison.Ordinal);
        Assert.Contains("internal SafeTensorRtObjectHandle Handle => _handle;", core, StringComparison.Ordinal);

        foreach (string suffix in new[] { string.Empty, ".InterfaceMetadata", ".Diagnostics", ".Lifecycle" })
        {
            Assert.DoesNotContain(
                "private sealed class CallbackState",
                ReadSource($"TensorRtProgressMonitor{suffix}.cs"),
                StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EvidenceConsumersEnumerateAllThreeCompleteSourceSets()
    {
        string readiness = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimePackageReadiness.ps1"));
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string[] sourceFiles =
        {
            "TensorRtProgressMonitor.cs",
            "TensorRtProgressMonitor.InterfaceMetadata.cs",
            "TensorRtProgressMonitor.Diagnostics.cs",
            "TensorRtProgressMonitor.Lifecycle.cs",
            "TensorRtProgressMonitor.Trampoline.cs",
            "TensorRtProgressMonitorEventKind.cs",
            "TensorRtProgressMonitorEvent.cs",
            "TensorRtProgressMonitorDiagnosticResult.cs",
            "TensorRtProgressMonitorHandler.cs",
            "TensorRtErrorRecorderDiagnosticsDesignGate.cs",
            "TensorRtErrorRecorderDiagnosticsDesignGateResult.cs",
            "TensorRtStreamIoInterfaceInfoDesignGate.cs",
            "TensorRtStreamIoInterfaceInfoDesignGateResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        foreach (string consumerFile in new[]
                 {
                     "ManagedProgressMonitorBoundaryTests.cs",
                     "PluginCreatorApiLanguageReadonlyTests.cs",
                     "VersionedInterfaceApiLanguageReadonlyUpliftTests.cs",
                     "ErrorRecorderDiagnosticsDesignGateTests.cs",
                     "ErrorRecorderSnapshotSummaryTests.cs",
                     "StreamIoInterfaceInfoDesignGateTests.cs"
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
    public void ProgressMonitorFilesRecomposeTheOriginalSource()
    {
        StringBuilder source = new(ProgressMonitorOriginalUsingHeader);
        foreach ((string fileName, string marker) in new[]
                 {
                     ("TensorRtProgressMonitorEventKind.cs", "/// <summary>"),
                     ("TensorRtProgressMonitorEvent.cs", "/// <summary>"),
                     ("TensorRtProgressMonitorDiagnosticResult.cs", "/// <summary>"),
                     ("TensorRtProgressMonitorHandler.cs", "/// <summary>")
                 })
        {
            source.Append(ReadTopLevelSegment(fileName, marker));
            source.Append('\n');
        }

        string core = Normalize(ReadSource("TensorRtProgressMonitor.cs"));
        int ownerStart = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class TensorRtProgressMonitor", ownerStart, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(ownerStart >= 0 && declaration > ownerStart && bodyStart > declaration);
        source.Append(core[ownerStart..(bodyStart + 1)].Replace(
            "public sealed partial class TensorRtProgressMonitor",
            "public sealed class TensorRtProgressMonitor",
            StringComparison.Ordinal));
        source.Append('\n');

        string coreBody = ReadPartialBody("TensorRtProgressMonitor.cs", "TensorRtProgressMonitor");
        int lineProperty = FindMemberStart(coreBody, "    public TensorRtApiLine Line");
        string[] lifecycle = ReadMemberSegments(
            "TensorRtProgressMonitor.Lifecycle.cs",
            "TensorRtProgressMonitor",
            "    ~TensorRtProgressMonitor()",
            "    public void Dispose(",
            "    internal void AttachBorrower(",
            "    internal void DetachBorrower(",
            "    internal void ThrowIfDisposed(",
            "    private void ReleaseHandle(",
            "    private void FreeCallbackState(");
        string[] trampoline = ReadMemberSegments(
            "TensorRtProgressMonitor.Trampoline.cs",
            "TensorRtProgressMonitor",
            "    private static BridgeStatusCode InvokeManagedProgressMonitor(",
            "    private static string DecodeUtf8(",
            "    private static string? DecodeUtf8Nullable(",
            "    private sealed class CallbackState");

        source.Append(coreBody[..lineProperty]);
        source.Append(lifecycle[0]);
        source.Append(coreBody[lineProperty..]);
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtProgressMonitor.InterfaceMetadata.cs", "TensorRtProgressMonitor"));
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtProgressMonitor.Diagnostics.cs", "TensorRtProgressMonitor"));
        source.Append('\n');
        source.Append(lifecycle[1]);
        source.Append(lifecycle[2]);
        source.Append(lifecycle[3]);
        source.Append(lifecycle[4]);
        source.Append(trampoline[0]);
        source.Append(trampoline[1]);
        source.Append(trampoline[2]);
        source.Append(lifecycle[5]);
        source.Append(lifecycle[6]);
        source.Append('\n');
        source.Append(trampoline[3]);
        source.Append("}\n");

        Assert.Equal(ProgressMonitorOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Theory]
    [InlineData(
        "TensorRtErrorRecorderDiagnosticsDesignGate.cs",
        "TensorRtErrorRecorderDiagnosticsDesignGateResult.cs",
        ErrorRecorderGateOriginalNormalizedSha256)]
    [InlineData(
        "TensorRtStreamIoInterfaceInfoDesignGate.cs",
        "TensorRtStreamIoInterfaceInfoDesignGateResult.cs",
        StreamIoGateOriginalNormalizedSha256)]
    public void GateAndResultFilesRecomposeTheOriginalSource(
        string gateFileName,
        string resultFileName,
        string expectedSha256)
    {
        StringBuilder source = new(Normalize(ReadSource(gateFileName)));
        source.Append('\n');
        source.Append(ReadTopLevelSegment(resultFileName, "/// <summary>"));
        Assert.Equal(expectedSha256, ComputeSha256(source.ToString()));
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
