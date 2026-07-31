using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCallbackOwnerClosureLayoutTests
{
    private const string OwnerOriginalNormalizedSha256 =
        "2f83d2a3b83480ab4653b085622f9458566448a43e08e8d62d5b97571257c9db";
    private const string MatrixOriginalNormalizedSha256 =
        "85a441af723e7f924ec5d3534d61353b56553465e17efb349fac56b79e36869f";

    public static TheoryData<string, string[]> OwnerFileMethods => new()
    {
        { "TensorRtDebugListenerCallbackOwner.cs", Array.Empty<string>() },
        { "TensorRtDebugListenerCallbackOwner.DesignDiagnostic.cs", new[] { "RunDesignDiagnostic" } },
        { "TensorRtDebugListenerCallbackOwner.Snapshots.cs", new[] { "GetSnapshot", "CreateSnapshot" } },
        {
            "TensorRtDebugListenerCallbackOwner.Lifecycle.cs",
            new[]
            {
                "Dispose", "AttachBorrower", "DetachBorrower", "ThrowIfDisposed", "ReleaseResources",
                "FreeCallbackState"
            }
        },
        { "TensorRtDebugListenerCallbackOwner.ShapeFormatting.cs", new[] { "GetDimension", "FormatShape" } },
        {
            "TensorRtDebugListenerCallbackOwner.Trampoline.cs",
            new[]
            {
                "InvokeDebugListenerDesignGate", "EnterCallback", "ExitCallback", "RecordInvocation", "RecordRequest",
                "RecordStatus", "RecordReturnedFailure", "RecordFailure", "RecordReleaseHook"
            }
        }
    };

    public static TheoryData<string, string[]> MatrixFileMethods => new()
    {
        { "TensorRtCallbackOwnerClosureMatrix.cs", new[] { "Evaluate" } },
        {
            "TensorRtCallbackOwnerClosureMatrix.Allocators.cs",
            new[] { "BuildGpuAllocatorRow", "BuildGpuAsyncAllocatorRow" }
        },
        {
            "TensorRtCallbackOwnerClosureMatrix.OutputDebug.cs",
            new[] { "BuildOutputAllocatorRow", "BuildDebugListenerRow" }
        },
        { "TensorRtCallbackOwnerClosureMatrix.StreamIo.cs", new[] { "BuildStreamReaderWriterRow" } },
        { "TensorRtCallbackOwnerClosureMatrix.RowConstruction.cs", new[] { "CreateRow" } },
        {
            "TensorRtCallbackOwnerClosureMatrix.Blockers.cs",
            new[] { "BuildMatrixBlockers", "AddBlockerIfFalse", "AddBlocker" }
        }
    };

    public static TheoryData<string, string, string[]> ModelProperties => new()
    {
        {
            "Debugging",
            "TensorRtDebugListenerCallbackRequest.cs",
            new[]
            {
                "TensorName", "DataType", "Location", "ShapeRank", "ShapeDimensions", "Reason", "IsInput",
                "IsOutput", "IsShapeTensor", "IsExecutionTensor"
            }
        },
        {
            "Debugging",
            "TensorRtDebugListenerCallbackOwnerSnapshot.cs",
            new[]
            {
                "EvidenceKind", "CallbackKind", "RuntimeEvidenceKind", "RealCallbackRuntime",
                "IsRealCallbackRuntimeProof", "OwnerId", "Operation", "Line", "LastStatus", "TensorName", "DataType",
                "Location", "ShapeRank", "ShapeSummary", "IsInput", "IsOutput", "IsShapeTensor", "IsExecutionTensor",
                "InvocationCount", "ProcessDebugTensorCount", "FailureCount", "InFlightCallbackCount",
                "MaxInFlightCallbackCount", "ActiveGateCallCount", "ReleaseHookCount", "CallbackStatePinned",
                "DelegatePinned", "DisposeRequested", "IsAttached", "DebugTensorMetadataCopied",
                "DebugTensorPointerExposed", "DebugTensorPointerProduced", "BorrowedDebugTensorPointerEscaped",
                "LastDiagnostic", "ReleaseDiagnostic", "Succeeded"
            }
        },
        {
            "Core",
            "TensorRtCallbackOwnerClosureMatrixRow.cs",
            new[]
            {
                "OwnerFamily", "CallbackKind", "CallbackMethods", "SupportedLines", "EvidenceKind", "RuntimeEvidenceKind",
                "DesignGateReady", "ManagedOwnerStateReady", "SafeHandleOrGcHandleKeepAliveReady",
                "NativeNonCopyableOwnerStorageReady", "NativeCreateDestroySymmetricReady", "AttachDetachClearControlReady",
                "DetachBeforeReleaseReady", "NoThrowDestructorReady", "NoThrowVTableReady",
                "ManagedExceptionCaptureReady", "ExceptionToStatusMappingReady", "InFlightCallbackAccountingReady",
                "BorrowedPointerEscapeBlocked", "OptInRuntimeSmokeReady", "PackageConsumerRuntimeProofRequired",
                "PackageConsumerRuntimeProofReady", "ClosureReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "NextWorkItem", "BlockedPrerequisites", "BlockedPrerequisiteCount",
                "ReadyClosureColumnCount", "TotalClosureColumnCount", "Status", "Diagnostic"
            }
        },
        {
            "Core",
            "TensorRtCallbackOwnerClosureMatrixResult.cs",
            new[]
            {
                "EvidenceKind", "RuntimeEvidenceKind", "RealCallbackRuntime", "IsRealCallbackRuntimeProof", "Rows",
                "FamilyCount", "DesignGateReadyFamilyCount", "ClosureReadyFamilyCount",
                "RuntimeProofAttemptReadyFamilyCount", "PackageConsumerRuntimeProofReadyFamilyCount",
                "PointerFreeSurfaceReady", "AllFamiliesClosureReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked",
                "DeferredRowsStillRequired", "BlockedPrerequisites", "BlockedPrerequisiteCount", "Status", "Summary"
            }
        }
    };

    [Theory]
    [MemberData(nameof(OwnerFileMethods))]
    public void OwnerPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource("Debugging", fileName)));
    }

    [Theory]
    [MemberData(nameof(MatrixFileMethods))]
    public void MatrixPartialsOwnExactMethods(string fileName, string[] expectedMethods)
    {
        Assert.Equal(expectedMethods, EnumerateMethodNames(ReadSource("Core", fileName)));
    }

    [Theory]
    [MemberData(nameof(ModelProperties))]
    public void ModelFilesOwnOneTopLevelTypeAndExactProperties(
        string module,
        string fileName,
        string[] expectedProperties)
    {
        string source = ReadSource(module, fileName);
        Assert.Single(Regex.Matches(
            source,
            @"^(?:public|internal)\s+(?:sealed\s+|readonly\s+)?(?:class|struct)\s+",
            RegexOptions.Multiline));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void CallbackStateAndDelegateStayWithTheTrampoline()
    {
        string trampoline = ReadSource("Debugging", "TensorRtDebugListenerCallbackOwner.Trampoline.cs");
        Assert.Single(Regex.Matches(trampoline, @"^    private sealed class CallbackState$", RegexOptions.Multiline));
        Assert.Single(Regex.Matches(
            trampoline,
            @"^    private delegate BridgeStatusCode TensorRtDebugListenerDesignGateCallback\(",
            RegexOptions.Multiline));

        foreach (string fileName in OwnerFileMethods.Select(row => row[0]).Cast<string>()
                     .Where(fileName => !string.Equals(
                         fileName,
                         "TensorRtDebugListenerCallbackOwner.Trampoline.cs",
                         StringComparison.Ordinal)))
        {
            string source = ReadSource("Debugging", fileName);
            Assert.DoesNotContain("private sealed class CallbackState", source, StringComparison.Ordinal);
            Assert.DoesNotContain("TensorRtDebugListenerDesignGateCallback(", source, StringComparison.Ordinal);
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
            "TensorRtDebugListenerCallbackOwner.cs",
            "TensorRtDebugListenerCallbackOwner.DesignDiagnostic.cs",
            "TensorRtDebugListenerCallbackOwner.Snapshots.cs",
            "TensorRtDebugListenerCallbackOwner.Lifecycle.cs",
            "TensorRtDebugListenerCallbackOwner.Trampoline.cs",
            "TensorRtDebugListenerCallbackOwner.ShapeFormatting.cs",
            "TensorRtDebugListenerCallbackRequest.cs",
            "TensorRtDebugListenerCallbackOwnerSnapshot.cs",
            "TensorRtCallbackOwnerClosureMatrix.cs",
            "TensorRtCallbackOwnerClosureMatrix.Allocators.cs",
            "TensorRtCallbackOwnerClosureMatrix.OutputDebug.cs",
            "TensorRtCallbackOwnerClosureMatrix.StreamIo.cs",
            "TensorRtCallbackOwnerClosureMatrix.RowConstruction.cs",
            "TensorRtCallbackOwnerClosureMatrix.Blockers.cs",
            "TensorRtCallbackOwnerClosureMatrixRow.cs",
            "TensorRtCallbackOwnerClosureMatrixResult.cs"
        };

        foreach (string sourceFile in sourceFiles)
        {
            Assert.Contains(sourceFile, readiness, StringComparison.Ordinal);
            Assert.Contains(sourceFile, testReader, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DebugListenerOwnerFilesRecomposeTheOriginalSource()
    {
        string request = Normalize(ReadSource("Debugging", "TensorRtDebugListenerCallbackRequest.cs"));
        int firstType = request.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(firstType >= 0);
        StringBuilder source = new(request[..firstType]);
        source.Append(ReadTopLevelSegment("Debugging", "TensorRtDebugListenerCallbackRequest.cs", "/// <summary>"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Debugging", "TensorRtDebugListenerCallbackOwnerSnapshot.cs", "/// <summary>"));
        source.Append('\n');

        string core = Normalize(ReadSource("Debugging", "TensorRtDebugListenerCallbackOwner.cs"));
        int ownerStart = core.IndexOf("/// <summary>", StringComparison.Ordinal);
        int declaration = core.IndexOf("public sealed partial class TensorRtDebugListenerCallbackOwner", ownerStart, StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(ownerStart >= 0 && declaration > ownerStart && bodyStart > declaration);
        source.Append(core[ownerStart..(bodyStart + 1)].Replace(
            "public sealed partial class TensorRtDebugListenerCallbackOwner",
            "public sealed class TensorRtDebugListenerCallbackOwner",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerCallbackOwner.cs", "TensorRtDebugListenerCallbackOwner"));

        string[] snapshots = ReadMemberSegments(
            "TensorRtDebugListenerCallbackOwner.Snapshots.cs",
            "    public TensorRtDebugListenerCallbackOwnerSnapshot GetSnapshot(",
            "    private TensorRtDebugListenerCallbackOwnerSnapshot CreateSnapshot(");
        string[] lifecycle = ReadMemberSegments(
            "TensorRtDebugListenerCallbackOwner.Lifecycle.cs",
            "    public void Dispose(",
            "    private void FreeCallbackState(");
        string[] shapes = ReadMemberSegments(
            "TensorRtDebugListenerCallbackOwner.ShapeFormatting.cs",
            "    private static long GetDimension(",
            "    private static string FormatShape(");
        string[] trampoline = ReadMemberSegments(
            "TensorRtDebugListenerCallbackOwner.Trampoline.cs",
            "    private static BridgeStatusCode InvokeDebugListenerDesignGate(",
            "    private sealed class CallbackState",
            "    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]");

        source.Append(ReadPartialBody("Debugging", "TensorRtDebugListenerCallbackOwner.DesignDiagnostic.cs", "TensorRtDebugListenerCallbackOwner"));
        source.Append(snapshots[0]);
        source.Append(lifecycle[0]);
        source.Append(lifecycle[1]);
        source.Append(snapshots[1]);
        source.Append(trampoline[0]);
        source.Append(shapes[0]);
        source.Append(shapes[1]);
        source.Append(trampoline[1]);
        source.Append(trampoline[2]);
        source.Append("}\n");

        Assert.Equal(OwnerOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void ClosureMatrixFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Core", "TensorRtCallbackOwnerClosureMatrix.cs"));
        int declaration = core.IndexOf("public static partial class TensorRtCallbackOwnerClosureMatrix", StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declaration);
        Assert.True(declaration >= 0 && bodyStart > declaration);

        StringBuilder source = new();
        source.Append(core[..(bodyStart + 1)].Replace(
            "public static partial class TensorRtCallbackOwnerClosureMatrix",
            "public static class TensorRtCallbackOwnerClosureMatrix",
            StringComparison.Ordinal));
        source.Append('\n');
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.Allocators.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.OutputDebug.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.StreamIo.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.RowConstruction.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append(ReadPartialBody("Core", "TensorRtCallbackOwnerClosureMatrix.Blockers.cs", "TensorRtCallbackOwnerClosureMatrix"));
        source.Append("}\n\n");
        source.Append(ReadTopLevelSegment("Core", "TensorRtCallbackOwnerClosureMatrixRow.cs", "/// <summary>"));
        source.Append('\n');
        source.Append(ReadTopLevelSegment("Core", "TensorRtCallbackOwnerClosureMatrixResult.cs", "/// <summary>"));

        Assert.Equal(MatrixOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string[] ReadMemberSegments(string fileName, params string[] declarationMarkers)
    {
        string body = ReadPartialBody("Debugging", fileName, "TensorRtDebugListenerCallbackOwner");
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
