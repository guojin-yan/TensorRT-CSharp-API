using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCallbackPrecheckAllocatorLayoutTests
{
    private const string PrecheckOriginalNormalizedSha256 =
        "27bef27bfec81e801af8663460c88bcefe43411a5ab4c3dbaebf14e907d7dab7";
    private const string AllocatorOriginalNormalizedSha256 =
        "dd04372ac040a7427aeec06c7cf8d86c286b7a5146d1dcd8ff27905a2cfe6945";

    private static readonly string[] PrecheckPrerequisiteTypes =
    {
        "TensorRtDebugListenerCallbackOwnerSnapshot",
        "TensorRtDebugListenerAttachDetachDesignGateResult",
        "TensorRtDebugListenerBorrowedTensorSafetyGateResult",
        "TensorRtDebugListenerAttachVTableSafetyGateResult",
        "TensorRtDebugListenerNativeAttachNoThrowPreflightResult",
        "TensorRtDebugListenerNativeOwnerAddressDesignGateResult",
        "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult",
        "TensorRtDebugListenerNativeAttachEntryDesignGateResult",
        "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult",
        "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult",
        "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult",
        "TensorRtDebugListenerNativeOwnerStableIdentityResult",
        "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult",
        "TensorRtDebugListenerNativeNoThrowDestructorResult",
        "TensorRtDebugListenerNativeOwnerLifecycleGateResult",
        "TensorRtDebugListenerNativeAttachBridgeShapeGateResult",
        "TensorRtDebugListenerExceptionStatusMappingGateResult",
        "TensorRtDebugListenerInFlightAccountingGateResult",
        "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult"
    };

    public static TheoryData<string, int[]> PrecheckFileParameterCounts => new()
    {
        { "TensorRtDebugListenerRuntimeProofPrecheck.DesignPrerequisites.cs", new[] { 1, 2, 3, 4 } },
        { "TensorRtDebugListenerRuntimeProofPrecheck.NativeAttachDesign.cs", new[] { 5, 6, 7, 8 } },
        { "TensorRtDebugListenerRuntimeProofPrecheck.OwnerLifecycle.cs", new[] { 9, 10 } },
        { "TensorRtDebugListenerRuntimeProofPrecheck.RuntimeScaffold.cs", new[] { 11, 12 } },
        { "TensorRtDebugListenerRuntimeProofPrecheck.FinalRuntimeGates.cs", new[] { 13, 14, 15 } },
        { "TensorRtDebugListenerRuntimeProofPrecheck.cs", new[] { 19 } }
    };

    public static TheoryData<string, string[]> AllocatorOwnerFileMethods => new()
    {
        { "TensorRtAllocatorCallbackOwner.cs", Array.Empty<string>() },
        {
            "TensorRtAllocatorCallbackOwner.Lifecycle.cs",
            new[] { "RunLifecycleDiagnostic", "GetSnapshot", "Dispose", "ThrowIfDisposed", "FreeCallbackState" }
        },
        { "TensorRtAllocatorCallbackOwner.ManagedDryRun.cs", new[] { "RunDryRunDiagnostic" } },
        { "TensorRtAllocatorCallbackOwner.NativeDryRun.cs", new[] { "RunNativeDryRunDiagnostic" } },
        { "TensorRtAllocatorCallbackOwner.StateLedger.cs", new[] { "RunNativeStateLedgerDryRunDiagnostic" } },
        {
            "TensorRtAllocatorCallbackOwner.InternalPrototype.cs",
            new[]
            {
                "RunInternalSyncAllocatorRuntimePrototype", "GetInternalRuntimePrototypeSnapshot",
                "CreateInternalRuntimePrototypeResult", "InvokeInternalSyncAllocatorPrototype", "EnterCallback",
                "ExitCallback", "RecordInvocation", "RecordDiagnostic", "RecordStatus", "RecordReturnedFailure",
                "RecordFailure", "RecordFailure", "RecordReleaseHook"
            }
        },
        {
            "TensorRtAllocatorCallbackOwner.ResultMapping.cs",
            new[] { "CreateNativeDryRunResult", "CreateStateDryRunResult" }
        }
    };

    public static TheoryData<string, string, string[]> AllocatorModelProperties => new()
    {
        {
            "TensorRtAllocatorDryRunRequest.cs",
            "TensorRtAllocatorDryRunRequest",
            new[] { "Size", "Alignment", "Reason" }
        },
        {
            "TensorRtAllocatorDryRunResult.cs",
            "TensorRtAllocatorDryRunResult",
            new[] { "Succeeded", "Diagnostic" }
        },
        {
            "TensorRtAllocatorNativeDryRunResult.cs",
            "TensorRtAllocatorNativeDryRunResult",
            new[] { "Line", "InvocationCount", "FailureCount", "LastStatus", "IsAttached", "LastSize", "LastAlignment", "Diagnostic", "Succeeded" }
        },
        {
            "TensorRtAllocatorOwnerStateDryRunResult.cs",
            "TensorRtAllocatorOwnerStateDryRunResult",
            new[]
            {
                "Line", "OwnerId", "StateTransitionCount", "LedgerAllocationCount", "LedgerReleaseCount",
                "LedgerFailureCount", "LastAllocationId", "LastReleaseAllocationId", "LastSize", "LastAlignment",
                "LastStreamValue", "AttachState", "LastStatus", "IsAttached", "HasLiveAllocation",
                "LastOperation", "Diagnostic", "Succeeded"
            }
        },
        {
            "TensorRtAllocatorCallbackOwnerSnapshot.cs",
            "TensorRtAllocatorCallbackOwnerSnapshot",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "OwnerId", "Operation", "LastStatus", "InvocationCount",
                "FailureCount", "InFlightCallbackCount", "MaxInFlightCallbackCount", "ActivePrototypeCallCount",
                "ReleaseHookCount", "CallbackStatePinned", "DelegatePinned", "DisposeRequested", "IsAttached",
                "DevicePointerExposed", "DevicePointerProduced", "BorrowedPointerEscaped", "LastDiagnostic",
                "ReleaseDiagnostic", "ManagedKeepAliveReady", "DisposeReleaseReady", "PointerFreeSurfaceReady", "Succeeded"
            }
        },
        {
            "TensorRtAllocatorInternalRuntimePrototypeResult.cs",
            "TensorRtAllocatorInternalRuntimePrototypeResult",
            new[]
            {
                "RealCallbackRuntime", "EvidenceKind", "CallbackKind", "OwnerId", "Operation", "LastStatus",
                "InvocationCount", "FailureCount", "InFlightCallbackCount", "MaxInFlightCallbackCount",
                "ActivePrototypeCallCount", "ReleaseHookCount", "CallbackStatePinned", "DelegatePinned",
                "DisposeRequested", "IsAttached", "LastDiagnostic", "ReleaseDiagnostic", "Succeeded"
            }
        }
    };

    [Theory]
    [MemberData(nameof(PrecheckFileParameterCounts))]
    public void PrecheckFilesOwnExactPrerequisiteSignatures(string fileName, int[] expectedCounts)
    {
        string[][] signatures = EnumerateEvaluateParameterTypes(ReadDebuggingSource(fileName));
        Assert.Equal(expectedCounts, signatures.Select(signature => signature.Length));
        foreach (string[] signature in signatures)
        {
            Assert.Equal(PrecheckPrerequisiteTypes.Take(signature.Length), signature);
        }
    }

    [Theory]
    [MemberData(nameof(AllocatorOwnerFileMethods))]
    public void AllocatorOwnerPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadAllocatorSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(AllocatorModelProperties))]
    public void AllocatorModelsOwnOneTypeAndExactProperties(
        string fileName,
        string expectedType,
        string[] expectedProperties)
    {
        string source = ReadAllocatorSource(fileName);
        MatchCollection declarations = Regex.Matches(
            source,
            @"^(?:public|internal)\s+(?:readonly\s+)?(?:struct|class)\s+(?<name>\w+)",
            RegexOptions.Multiline);
        Assert.Single(declarations);
        Assert.Equal(expectedType, declarations[0].Groups["name"].Value);
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void AllocatorDelegateOwnsOnePublicDelegate()
    {
        string source = ReadAllocatorSource("TensorRtAllocatorDryRunHandler.cs");
        Assert.Single(Regex.Matches(source, @"^public delegate ", RegexOptions.Multiline));
        Assert.Contains(
            "public delegate TensorRtAllocatorDryRunResult TensorRtAllocatorDryRunHandler(TensorRtAllocatorDryRunRequest request);",
            source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceConsumersEnumerateTheCompleteSplitSourceSets()
    {
        string readiness = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimePackageReadiness.ps1"));
        string testReader = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "JYPPX.ProjectQuality.Tests",
            "RepositorySourceReader.cs"));
        string[] sourceFiles =
        {
            "TensorRtDebugListenerRuntimeProofPrecheck.cs",
            "TensorRtDebugListenerRuntimeProofPrecheck.DesignPrerequisites.cs",
            "TensorRtDebugListenerRuntimeProofPrecheck.NativeAttachDesign.cs",
            "TensorRtDebugListenerRuntimeProofPrecheck.OwnerLifecycle.cs",
            "TensorRtDebugListenerRuntimeProofPrecheck.RuntimeScaffold.cs",
            "TensorRtDebugListenerRuntimeProofPrecheck.FinalRuntimeGates.cs",
            "TensorRtDebugListenerRuntimeProofPrecheckResult.cs",
            "TensorRtAllocatorCallbackOwner.cs",
            "TensorRtAllocatorCallbackOwner.Lifecycle.cs",
            "TensorRtAllocatorCallbackOwner.ManagedDryRun.cs",
            "TensorRtAllocatorCallbackOwner.NativeDryRun.cs",
            "TensorRtAllocatorCallbackOwner.StateLedger.cs",
            "TensorRtAllocatorCallbackOwner.InternalPrototype.cs",
            "TensorRtAllocatorCallbackOwner.ResultMapping.cs",
            "TensorRtAllocatorDryRunRequest.cs",
            "TensorRtAllocatorDryRunResult.cs",
            "TensorRtAllocatorDryRunHandler.cs",
            "TensorRtAllocatorNativeDryRunResult.cs",
            "TensorRtAllocatorOwnerStateDryRunResult.cs",
            "TensorRtAllocatorCallbackOwnerSnapshot.cs",
            "TensorRtAllocatorInternalRuntimePrototypeResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }

        Assert.Equal(28, Regex.Matches(readiness, @"Get-EvidenceSourceText -Path \$path").Count);
        Assert.DoesNotContain(
            "$combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8)",
            readiness,
            StringComparison.Ordinal);
    }

    [Fact]
    public void PrecheckFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadDebuggingSource("TensorRtDebugListenerRuntimeProofPrecheck.cs"));
        int declaration = core.IndexOf("public static partial class TensorRtDebugListenerRuntimeProofPrecheck", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public static partial class TensorRtDebugListenerRuntimeProofPrecheck",
            "public static class TensorRtDebugListenerRuntimeProofPrecheck",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.DesignPrerequisites.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.NativeAttachDesign.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.OwnerLifecycle.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.RuntimeScaffold.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.FinalRuntimeGates.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs", "TensorRtDebugListenerRuntimeProofPrecheck"));
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment("Debugging", "TensorRtDebugListenerRuntimeProofPrecheckResult.cs", "/// <summary>"));

        Assert.Equal(PrecheckOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void AllocatorFilesRecomposeTheOriginalSource()
    {
        string request = Normalize(ReadAllocatorSource("TensorRtAllocatorDryRunRequest.cs"));
        int firstType = request.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(firstType >= 0);

        StringBuilder source = new(request[..firstType]);
        foreach (string fileName in new[]
                 {
                     "TensorRtAllocatorDryRunRequest.cs",
                     "TensorRtAllocatorDryRunResult.cs",
                     "TensorRtAllocatorDryRunHandler.cs",
                     "TensorRtAllocatorNativeDryRunResult.cs",
                     "TensorRtAllocatorOwnerStateDryRunResult.cs",
                     "TensorRtAllocatorCallbackOwnerSnapshot.cs"
                 })
        {
            source.Append(ReadTopLevelSegment("MemoryAllocation", fileName, "/// <summary>"));
            source.Append('\n');
        }

        string ownerCore = Normalize(ReadAllocatorSource("TensorRtAllocatorCallbackOwner.cs"));
        int ownerStart = ownerCore.IndexOf("/// <summary>", StringComparison.Ordinal);
        int ownerDeclaration = ownerCore.IndexOf("public sealed partial class TensorRtAllocatorCallbackOwner", ownerStart, StringComparison.Ordinal);
        int ownerBodyStart = ownerCore.IndexOf('{', ownerDeclaration);
        Assert.True(ownerStart >= 0 && ownerDeclaration > ownerStart && ownerBodyStart > ownerDeclaration);
        source.Append(ownerCore[ownerStart..(ownerBodyStart + 1)].Replace(
            "public sealed partial class TensorRtAllocatorCallbackOwner",
            "public sealed class TensorRtAllocatorCallbackOwner",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("MemoryAllocation", "TensorRtAllocatorCallbackOwner.cs", "TensorRtAllocatorCallbackOwner"));

        string[] lifecycle = ReadMemberSegments(
            "TensorRtAllocatorCallbackOwner.Lifecycle.cs",
            "    public TensorRtAllocatorCallbackOwnerSnapshot RunLifecycleDiagnostic(",
            "    public TensorRtAllocatorCallbackOwnerSnapshot GetSnapshot(",
            "    public void Dispose(",
            "    private void ThrowIfDisposed(",
            "    private void FreeCallbackState(");
        string[] internalPrototype = ReadMemberSegments(
            "TensorRtAllocatorCallbackOwner.InternalPrototype.cs",
            "    internal TensorRtAllocatorInternalRuntimePrototypeResult RunInternalSyncAllocatorRuntimePrototype(",
            "    internal TensorRtAllocatorInternalRuntimePrototypeResult GetInternalRuntimePrototypeSnapshot(",
            "    private TensorRtAllocatorInternalRuntimePrototypeResult CreateInternalRuntimePrototypeResult(",
            "    private static BridgeStatusCode InvokeInternalSyncAllocatorPrototype(",
            "    private sealed class CallbackState",
            "    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]");
        string[] resultMapping = ReadMemberSegments(
            "TensorRtAllocatorCallbackOwner.ResultMapping.cs",
            "    private static TensorRtAllocatorNativeDryRunResult CreateNativeDryRunResult(",
            "    private static TensorRtAllocatorOwnerStateDryRunResult CreateStateDryRunResult(");

        source.Append(lifecycle[0]);
        source.Append(lifecycle[1]);
        source.Append(internalPrototype[0]);
        source.Append(internalPrototype[1]);
        source.Append(ReadPartialBody("MemoryAllocation", "TensorRtAllocatorCallbackOwner.ManagedDryRun.cs", "TensorRtAllocatorCallbackOwner"));
        source.Append(ReadPartialBody("MemoryAllocation", "TensorRtAllocatorCallbackOwner.NativeDryRun.cs", "TensorRtAllocatorCallbackOwner"));
        source.Append(ReadPartialBody("MemoryAllocation", "TensorRtAllocatorCallbackOwner.StateLedger.cs", "TensorRtAllocatorCallbackOwner"));
        source.Append(lifecycle[2]);
        source.Append(lifecycle[3]);
        source.Append(lifecycle[4]);
        source.Append(internalPrototype[2]);
        source.Append(internalPrototype[3]);
        source.Append(resultMapping[0]);
        source.Append(resultMapping[1]);
        source.Append(internalPrototype[4]);
        source.Append(internalPrototype[5]);
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment(
            "MemoryAllocation",
            "TensorRtAllocatorInternalRuntimePrototypeResult.cs",
            "internal readonly struct TensorRtAllocatorInternalRuntimePrototypeResult"));

        Assert.Equal(AllocatorOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] ReadMemberSegments(string fileName, params string[] declarationMarkers)
    {
        string body = ReadPartialBody("MemoryAllocation", fileName, "TensorRtAllocatorCallbackOwner");
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
        if (declarationMarker.Contains("[UnmanagedFunctionPointer", StringComparison.Ordinal))
        {
            return declaration;
        }

        int summary = body.LastIndexOf("    /// <summary>", declaration, StringComparison.Ordinal);
        int previousMemberClose = body.LastIndexOf("\n    }", declaration, StringComparison.Ordinal);
        return summary > previousMemberClose ? summary : declaration;
    }

    private static string[][] EnumerateEvaluateParameterTypes(string source)
    {
        return Regex.Matches(
                source,
                @"public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate\((?<parameters>.*?)\)\s*\{",
                RegexOptions.Singleline)
            .Select(match => Regex.Matches(
                    match.Groups["parameters"].Value,
                    @"^\s*(?<type>TensorRt\w+)\s+\w+,?\s*$",
                    RegexOptions.Multiline)
                .Select(parameter => parameter.Groups["type"].Value)
                .ToArray())
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

    private static string ReadTopLevelSegment(string module, string fileName, string marker)
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

    private static string ReadDebuggingSource(string fileName) => ReadSource("Debugging", fileName);

    private static string ReadAllocatorSource(string fileName) => ReadSource("MemoryAllocation", fileName);

    private static string ReadSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            "Callbacks",
            module,
            fileName));
    }
}
