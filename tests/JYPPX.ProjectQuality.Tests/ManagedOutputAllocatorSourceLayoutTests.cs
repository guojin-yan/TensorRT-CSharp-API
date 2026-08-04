using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedOutputAllocatorSourceLayoutTests
{
    private const string RuntimeGateOriginalNormalizedSha256 =
        "d2d6e4a13731145ea77d01fe52b3b0538a3c12b1d8fdee9a0263599dd3fc3f13";
    private const string CallbackOwnerOriginalNormalizedSha256 =
        "885640c061761b8af11ac18f764b067320d3715769d5112055ace93876d2d655";

    public static TheoryData<string, string[]> RuntimeGateFileMethods => new()
    {
        { "TensorRtOutputAllocatorRuntimeGate.cs", Array.Empty<string>() },
        {
            "TensorRtOutputAllocatorRuntimeGate.Entries.cs",
            new[] { "RunInternalNotifyShapeRuntimeGate", "RunInternalReallocateOutputRuntimeGate" }
        },
        {
            "TensorRtOutputAllocatorRuntimeGate.Snapshots.cs",
            new[] { "GetInternalRuntimeGateSnapshot", "CreateResult" }
        },
        { "TensorRtOutputAllocatorRuntimeGate.Lifecycle.cs", new[] { "Dispose", "FreeCallbackState" } },
        { "TensorRtOutputAllocatorRuntimeGate.Invocation.cs", new[] { "RunInternalRuntimeGate" } },
        { "TensorRtOutputAllocatorRuntimeGate.Formatting.cs", new[] { "OperationName", "FormatShape" } },
        {
            "TensorRtOutputAllocatorRuntimeGate.Trampoline.cs",
            new[]
            {
                "InvokeOutputAllocatorRuntimeGate", "EnterCallback", "ExitCallback", "RecordInvocation",
                "RecordRequest", "RecordStatus", "RecordReturnedFailure", "RecordFailure", "RecordReleaseHook"
            }
        }
    };

    public static TheoryData<string, string[]> CallbackOwnerFileMethods => new()
    {
        { "TensorRtOutputAllocatorCallbackOwner.cs", Array.Empty<string>() },
        { "TensorRtOutputAllocatorCallbackOwner.DesignDiagnostic.cs", new[] { "RunDesignDiagnostic" } },
        { "TensorRtOutputAllocatorCallbackOwner.Snapshots.cs", new[] { "GetSnapshot" } },
        {
            "TensorRtOutputAllocatorCallbackOwner.Lifecycle.cs",
            new[] { "Dispose", "AttachBorrower", "DetachBorrower", "ThrowIfDisposed", "ReleaseResources", "FreeRuntimeCallbackHandles" }
        },
        {
            "TensorRtOutputAllocatorCallbackOwner.RuntimeCallback.cs",
            new[] { "InvokeManagedOutputAllocator", "DecodeRuntimeTensorName", "CopyRuntimeShape" }
        },
        { "TensorRtOutputAllocatorCallbackOwner.RuntimeSnapshot.cs", new[] { "GetRuntimeSnapshot" } }
    };

    public static TheoryData<string, string[]> ModelProperties => new()
    {
        {
            "TensorRtOutputAllocatorRuntimeGateRequest.cs",
            new[] { "TensorName", "RequestedSize", "Alignment", "ShapeRank", "Reason", "HasCurrentMemory" }
        },
        {
            "TensorRtOutputAllocatorRuntimeGateResult.cs",
            new[]
            {
                "RealCallbackRuntime", "EvidenceKind", "CallbackKind", "OwnerId", "Operation", "TensorName",
                "RequestedSize", "Alignment", "ShapeRank", "ShapeSummary", "HasCurrentMemory", "LastStatus",
                "InvocationCount", "NotifyShapeCount", "ReallocateOutputCount", "FailureCount", "InFlightCallbackCount",
                "MaxInFlightCallbackCount", "ActiveGateCallCount", "ReleaseHookCount", "CallbackStatePinned",
                "DelegatePinned", "DisposeRequested", "IsAttached", "OutputBufferPointerExposed",
                "OutputBufferPointerProduced", "LastDiagnostic", "ReleaseDiagnostic", "Succeeded"
            }
        },
        {
            "TensorRtOutputAllocatorCallbackRequest.cs",
            new[] { "Kind", "TensorName", "RequestedSize", "Alignment", "ShapeRank", "ShapeDimensions", "Reason", "HasCurrentMemory", "HasStream" }
        },
        {
            "TensorRtOutputAllocatorRuntimeSnapshot.cs",
            new[]
            {
                "Line", "OwnerId", "InvocationCount", "NotifyShapeCount", "ReallocateOutputCount", "FailureCount",
                "InFlightCallbackCount", "MaxInFlightCallbackCount", "AttachCount", "DetachCount", "AllocationCount",
                "ReuseCount", "ReleaseCount", "LiveAllocationCount", "LiveAllocationBytes", "PeakLiveAllocationBytes",
                "LastRequestedSize", "LastAlignment", "LastStatus", "IsAttached", "LastCallbackSucceeded",
                "LastAllocationSucceeded", "LastHadCurrentMemory", "LastHadStream", "LastCallbackKind", "TensorName",
                "ShapeDimensions", "Diagnostic", "NativePointerExposed", "RealCallbackRuntime", "RuntimeEvidenceKind"
            }
        },
        {
            "TensorRtOutputAllocatorCallbackOwnerSnapshot.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "Line", "OwnerId", "NativeOwnerId", "Operation", "TensorName",
                "RequestedSize", "Alignment", "ShapeRank", "ShapeSummary", "HasCurrentMemory", "LastStatus",
                "RuntimeGateStatus", "NativeLedgerStatus", "NativeLedgerAvailable", "InvocationCount", "NotifyShapeCount",
                "ReallocateOutputCount", "FailureCount", "InFlightCallbackCount", "MaxInFlightCallbackCount",
                "ActiveGateCallCount", "ReleaseHookCount", "CallbackStatePinned", "DelegatePinned", "DisposeRequested",
                "IsAttached", "StateTransitionCount", "LedgerAllocationCount", "LedgerReleaseCount", "LedgerFailureCount",
                "LastAllocationId", "LastReleaseAllocationId", "LastStreamValue", "HasLiveAllocation",
                "OutputBufferPointerExposed", "OutputBufferPointerProduced", "NativeLastOperation", "LastDiagnostic",
                "NativeLedgerDiagnostic", "ReleaseDiagnostic", "Succeeded"
            }
        }
    };

    [Theory]
    [MemberData(nameof(RuntimeGateFileMethods))]
    public void RuntimeGatePartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(CallbackOwnerFileMethods))]
    public void CallbackOwnerPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource(fileName)));
    }

    [Theory]
    [MemberData(nameof(ModelProperties))]
    public void ModelFilesOwnOneTopLevelTypeAndExactProperties(string fileName, string[] expectedProperties)
    {
        string source = ReadSource(fileName);
        Assert.Single(Regex.Matches(
            source,
            @"^(?:public|internal)\s+(?:sealed\s+|readonly\s+)?(?:class|struct)\s+",
            RegexOptions.Multiline));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void RuntimeCallbackStateAndDelegateStayWithTheTrampoline()
    {
        string trampoline = ReadSource("TensorRtOutputAllocatorRuntimeGate.Trampoline.cs");
        Assert.Single(Regex.Matches(trampoline, @"^    private sealed class CallbackState$", RegexOptions.Multiline));
        Assert.Single(Regex.Matches(
            trampoline,
            @"^    private delegate BridgeStatusCode TensorRtOutputAllocatorInternalRuntimeGateCallback\(",
            RegexOptions.Multiline));

        foreach (string fileName in new[]
                 {
                     "TensorRtOutputAllocatorRuntimeGate.cs",
                     "TensorRtOutputAllocatorRuntimeGate.Entries.cs",
                     "TensorRtOutputAllocatorRuntimeGate.Snapshots.cs",
                     "TensorRtOutputAllocatorRuntimeGate.Lifecycle.cs",
                     "TensorRtOutputAllocatorRuntimeGate.Invocation.cs",
                     "TensorRtOutputAllocatorRuntimeGate.Formatting.cs"
                 })
        {
            string source = ReadSource(fileName);
            Assert.DoesNotContain("private sealed class CallbackState", source, StringComparison.Ordinal);
            Assert.DoesNotContain("TensorRtOutputAllocatorInternalRuntimeGateCallback(", source, StringComparison.Ordinal);
        }
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
            "TensorRtOutputAllocatorRuntimeGate.cs",
            "TensorRtOutputAllocatorRuntimeGate.Entries.cs",
            "TensorRtOutputAllocatorRuntimeGate.Snapshots.cs",
            "TensorRtOutputAllocatorRuntimeGate.Lifecycle.cs",
            "TensorRtOutputAllocatorRuntimeGate.Invocation.cs",
            "TensorRtOutputAllocatorRuntimeGate.Trampoline.cs",
            "TensorRtOutputAllocatorRuntimeGate.Formatting.cs",
            "TensorRtOutputAllocatorRuntimeGateRequest.cs",
            "TensorRtOutputAllocatorRuntimeGateResult.cs",
            "TensorRtOutputAllocatorCallbackOwner.cs",
            "TensorRtOutputAllocatorCallbackOwner.DesignDiagnostic.cs",
            "TensorRtOutputAllocatorCallbackOwner.Snapshots.cs",
            "TensorRtOutputAllocatorCallbackOwner.Lifecycle.cs",
            "TensorRtOutputAllocatorCallbackOwner.RuntimeCallback.cs",
            "TensorRtOutputAllocatorCallbackOwner.RuntimeSnapshot.cs",
            "TensorRtOutputAllocatorCallbackRequest.cs",
            "TensorRtOutputAllocatorCallbackOwnerSnapshot.cs",
            "TensorRtOutputAllocatorCallbackKind.cs",
            "TensorRtOutputAllocatorHandler.cs",
            "TensorRtOutputAllocatorRuntimeSnapshot.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RuntimeGateFilesRecomposeTheOriginalSource()
    {
        string request = Normalize(ReadSource("TensorRtOutputAllocatorRuntimeGateRequest.cs"));
        int requestStart = request.IndexOf("internal readonly struct TensorRtOutputAllocatorRuntimeGateRequest", StringComparison.Ordinal);
        Assert.True(requestStart >= 0);
        StringBuilder source = new(request[..requestStart]);
        source.Append(request[requestStart..]);
        source.Append('\n');

        string core = Normalize(ReadSource("TensorRtOutputAllocatorRuntimeGate.cs"));
        int declaration = core.IndexOf("internal sealed partial class TensorRtOutputAllocatorRuntimeGate", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);
        source.Append(core[declaration..(bodyStart + 1)].Replace(
            "internal sealed partial class TensorRtOutputAllocatorRuntimeGate",
            "internal sealed class TensorRtOutputAllocatorRuntimeGate",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtOutputAllocatorRuntimeGate.cs", "TensorRtOutputAllocatorRuntimeGate"));

        string[] snapshots = ReadMemberSegments(
            "TensorRtOutputAllocatorRuntimeGate.Snapshots.cs",
            "TensorRtOutputAllocatorRuntimeGate",
            "    internal TensorRtOutputAllocatorRuntimeGateResult GetInternalRuntimeGateSnapshot(",
            "    private TensorRtOutputAllocatorRuntimeGateResult CreateResult(");
        string[] lifecycle = ReadMemberSegments(
            "TensorRtOutputAllocatorRuntimeGate.Lifecycle.cs",
            "TensorRtOutputAllocatorRuntimeGate",
            "    public void Dispose(",
            "    private void FreeCallbackState(");
        string[] formatting = ReadMemberSegments(
            "TensorRtOutputAllocatorRuntimeGate.Formatting.cs",
            "TensorRtOutputAllocatorRuntimeGate",
            "    private static string OperationName(",
            "    private static string FormatShape(");
        string[] trampoline = ReadMemberSegments(
            "TensorRtOutputAllocatorRuntimeGate.Trampoline.cs",
            "TensorRtOutputAllocatorRuntimeGate",
            "    private static BridgeStatusCode InvokeOutputAllocatorRuntimeGate(",
            "    private sealed class CallbackState",
            "    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]");

        source.Append(ReadPartialBody("TensorRtOutputAllocatorRuntimeGate.Entries.cs", "TensorRtOutputAllocatorRuntimeGate"));
        source.Append(snapshots[0]);
        source.Append(lifecycle[0]);
        source.Append(ReadPartialBody("TensorRtOutputAllocatorRuntimeGate.Invocation.cs", "TensorRtOutputAllocatorRuntimeGate"));
        source.Append(lifecycle[1]);
        source.Append(snapshots[1]);
        source.Append(trampoline[0]);
        source.Append(formatting[0]);
        source.Append(formatting[1]);
        source.Append(trampoline[1]);
        source.Append(trampoline[2]);
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment(
            "TensorRtOutputAllocatorRuntimeGateResult.cs",
            "internal readonly struct TensorRtOutputAllocatorRuntimeGateResult"));

        Assert.Equal(RuntimeGateOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CallbackOwnerFilesRecomposeTheOriginalSource()
    {
        string request = Normalize(ReadSource("TensorRtOutputAllocatorCallbackRequest.cs"));
        int firstType = request.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(firstType >= 0);
        StringBuilder source = new(request[..firstType]);
        source.Append(ReadTopLevelSegment("TensorRtOutputAllocatorCallbackRequest.cs", "/// <summary>"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("TensorRtOutputAllocatorCallbackOwnerSnapshot.cs", "/// <summary>"));
        source.Append('\n');

        string core = Normalize(ReadSource("TensorRtOutputAllocatorCallbackOwner.cs"));
        int ownerStart = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class TensorRtOutputAllocatorCallbackOwner", ownerStart, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(ownerStart >= 0 && declaration > ownerStart && bodyStart > declaration);
        source.Append(core[ownerStart..(bodyStart + 1)].Replace(
            "public sealed partial class TensorRtOutputAllocatorCallbackOwner",
            "public sealed class TensorRtOutputAllocatorCallbackOwner",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("TensorRtOutputAllocatorCallbackOwner.cs", "TensorRtOutputAllocatorCallbackOwner"));
        source.Append(ReadPartialBody("TensorRtOutputAllocatorCallbackOwner.DesignDiagnostic.cs", "TensorRtOutputAllocatorCallbackOwner"));
        source.Append(ReadPartialBody("TensorRtOutputAllocatorCallbackOwner.Snapshots.cs", "TensorRtOutputAllocatorCallbackOwner"));
        source.Append(ReadPartialBody("TensorRtOutputAllocatorCallbackOwner.Lifecycle.cs", "TensorRtOutputAllocatorCallbackOwner"));
        source.Append("}\n");

        Assert.Equal(CallbackOwnerOriginalNormalizedSha256, ComputeSha256(source.ToString()));
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
        if (declarationMarker.Contains("[UnmanagedFunctionPointer", StringComparison.Ordinal))
        {
            return declaration;
        }

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
            "MemoryAllocation",
            fileName));
    }
}
