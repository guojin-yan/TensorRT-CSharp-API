using System.Reflection;
using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class StreamIoInterfaceInfoDesignGateTests
{
    [Fact]
    public void StreamIoGateListsInterfaceInfoCandidatesWithoutExposingPointers()
    {
        TensorRtStreamIoInterfaceInfoDesignGateResult gate =
            TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("copied stream reader/writer interface metadata snapshot", gate.RequiredOutputMode);
        Assert.Contains("IStreamReader", gate.CandidateInterfaces);
        Assert.Contains("IStreamReaderV2", gate.CandidateInterfaces);
        Assert.Contains("IStreamWriter", gate.CandidateInterfaces);
        Assert.Contains("IStreamReader::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IStreamReaderV2::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IStreamWriter::getInterfaceInfo", gate.CandidateMethods);
        Assert.Equal(3, gate.CandidateMethodCount);
        Assert.Contains("IStreamReader::read", gate.DirectCallbackMethods);
        Assert.Contains("IStreamReaderV2::read", gate.DirectCallbackMethods);
        Assert.Contains("IStreamReaderV2::seek", gate.DirectCallbackMethods);
        Assert.Contains("IStreamWriter::write", gate.DirectCallbackMethods);
        Assert.Equal(4, gate.DirectCallbackMethodCount);
        Assert.Contains("IStreamReader::getAPILanguage", gate.ApiLanguageCandidateMethods);
        Assert.Contains("IStreamReaderV2::getAPILanguage", gate.ApiLanguageCandidateMethods);
        Assert.Contains("IStreamWriter::getAPILanguage", gate.ApiLanguageCandidateMethods);
        Assert.Contains("IVersionedInterface::getAPILanguage", gate.ApiLanguageCandidateMethods);
        Assert.Equal(4, gate.ApiLanguageCandidateMethodCount);
        Assert.Equal(11, gate.OwnerLedgerTrackedMethodCount);
        Assert.Contains("SafeHandle-derived TensorRtStreamReaderOwnerHandle", gate.OwnerHandleCandidates);
        Assert.Contains("GCHandle-backed managed stream owner state", gate.OwnerHandleCandidates);
        Assert.Contains("pin managed owner state before native attach and free it only after detach", gate.OwnerLifecycleRequirements);
        Assert.Contains("native vtable callbacks must be noexcept", gate.CallbackSafetyRequirements);
        Assert.Contains("reader/writer handles", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=3", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("DirectCallbackMethodCount=4", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("ApiLanguageCandidateMethodCount=4", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("OwnerLedgerTrackedMethodCount=11", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("ManagedOwnedStreamOwnerLedgerReady=True", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("ManagedOwnerLifetimeReady=False", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NativeOwnerCreateDestroySymmetric=False", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("NoThrowVTableReady=False", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("ExceptionToStatusMappingReady=False", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("DetachBeforeReleaseReady=False", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.ManagedOwnedStreamOwnerLedgerReady);
        Assert.False(gate.ManagedOwnerLifetimeReady);
        Assert.False(gate.NativeOwnerCreateDestroySymmetric);
        Assert.False(gate.NoThrowVTableReady);
        Assert.False(gate.ExceptionToStatusMappingReady);
        Assert.False(gate.DetachBeforeReleaseReady);
        Assert.False(gate.CanImplementStreamMetadataNow);
        Assert.False(gate.StreamCallbackBridgeReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.DirectStreamCallbackRowsDeferred);
        Assert.False(gate.StreamReaderPointerExposed);
        Assert.False(gate.StreamWriterPointerExposed);
        Assert.False(gate.StreamCallbackInvocationEnabled);
        Assert.False(gate.StreamBufferExposed);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void CandidateListRecordsStreamIoBoundary()
    {
        JsonElement candidate = FindReadonlyCandidate("stream-io-interface-info-design-004");

        Assert.Equal("design-gate-ready-not-runtime-proof", candidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("copied stream reader/writer interface metadata snapshot", candidate.GetProperty("outputMode").GetString());
        AssertCandidateMethods(
            candidate,
            "IStreamReader::getInterfaceInfo",
            "IStreamReaderV2::getInterfaceInfo",
            "IStreamWriter::getInterfaceInfo",
            "IStreamReader::read",
            "IStreamReaderV2::read",
            "IStreamReaderV2::seek",
            "IStreamWriter::write",
            "IStreamReader::getAPILanguage",
            "IStreamReaderV2::getAPILanguage",
            "IStreamWriter::getAPILanguage",
            "IVersionedInterface::getAPILanguage");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtStreamIoInterfaceInfoDesignGate.cs");
        AssertEvidenceContains(candidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/StreamIoInterfaceInfoDesignGateTests.cs");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtStreamIoInterfaceInfoDesignGateResult.OwnerLifecycleRequirements");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtStreamIoInterfaceInfoDesignGateResult.CallbackSafetyRequirements");

        string ownership = candidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("reader/writer handles are not exposed", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("read/seek/write callbacks remain deferred", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("SafeHandle/GCHandle lifetime", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no-throw vtable", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("exception-to-status mapping", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("detach-before-release ordering", ownership, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void StreamOwnerLedgerDocsRecordCallbackBlockers()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtStreamIoInterfaceInfoDesignGate.cs");
        string doc = ReadSource("docs", "articles", "zh-cn", "stream-io-interface-info-design-gate.md");

        Assert.Contains("ManagedOwnedStreamOwnerLedgerReady", source);
        Assert.Contains("OwnerHandleCandidates", source);
        Assert.Contains("OwnerLifecycleRequirements", source);
        Assert.Contains("CallbackSafetyRequirements", source);
        Assert.Contains("DirectCallbackMethods", source);
        Assert.Contains("ApiLanguageCandidateMethods", source);
        Assert.Contains("CanImplementStreamMetadataNow", source);
        Assert.Contains("StreamCallbackBridgeReady", source);

        Assert.Contains("IStreamReader::read", doc);
        Assert.Contains("IStreamReaderV2::seek", doc);
        Assert.Contains("IStreamWriter::write", doc);
        Assert.Contains("getAPILanguage", doc);
        Assert.Contains("SafeHandle", doc);
        Assert.Contains("GCHandle", doc);
        Assert.Contains("no-throw", doc);
        Assert.Contains("exception", doc);
        Assert.Contains("detach", doc);
        Assert.Contains("不是 runtime proof", doc);
    }

    [Fact]
    public void PublicStreamIoGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtStreamIoInterfaceInfoDesignGate),
            typeof(TensorRtStreamIoInterfaceInfoDesignGateResult)
        };

        foreach (Type type in publicTypes)
        {
            foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
            {
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }

            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }
        }
    }

    private static JsonElement FindReadonlyCandidate(string candidateId)
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement readonlyDiagnostics = document.RootElement.GetProperty("groups").GetProperty("readonlyDiagnostics");
        foreach (JsonElement candidate in readonlyDiagnostics.EnumerateArray())
        {
            if (candidate.GetProperty("candidateId").GetString() == candidateId)
            {
                return candidate.Clone();
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertCandidateMethods(JsonElement candidate, params string[] expectedMethods)
    {
        string[] actual = candidate.GetProperty("candidateMethods")
            .EnumerateArray()
            .Select(static item => item.GetString() ?? string.Empty)
            .ToArray();

        foreach (string expected in expectedMethods)
        {
            Assert.Contains(expected, actual);
        }
    }

    private static void AssertEvidenceContains(JsonElement candidate, string arrayName, string expected)
    {
        JsonElement evidence = candidate.GetProperty("implementationEvidence");
        foreach (JsonElement item in evidence.GetProperty(arrayName).EnumerateArray())
        {
            if (item.GetString() == expected)
            {
                return;
            }
        }

        throw new InvalidOperationException($"Expected {expected} in {candidate.GetProperty("candidateId").GetString()} evidence {arrayName}.");
    }
}
