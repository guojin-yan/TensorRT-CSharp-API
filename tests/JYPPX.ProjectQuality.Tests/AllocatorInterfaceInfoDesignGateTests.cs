using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class AllocatorInterfaceInfoDesignGateTests
{
    [Fact]
    public void AllocatorInterfaceInfoGateKeepsCallbackOwnershipDeferred()
    {
        TensorRtAllocatorInterfaceInfoDesignGateResult gate =
            TensorRtAllocatorInterfaceInfoDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("copied allocator interface metadata snapshot", gate.RequiredOutputMode);
        Assert.Contains("IGpuAllocator", gate.CandidateInterfaces);
        Assert.Contains("IGpuAsyncAllocator", gate.CandidateInterfaces);
        Assert.Contains("IOutputAllocator", gate.CandidateInterfaces);
        Assert.Contains("IGpuAllocator::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IGpuAsyncAllocator::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IOutputAllocator::getInterfaceInfo", gate.CandidateMethods);
        Assert.Equal(3, gate.CandidateMethodCount);
        Assert.Contains("direct allocator callbacks deferred", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=3", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.DirectAllocatorCallbackRowsDeferred);
        Assert.False(gate.AllocatorPointerExposed);
        Assert.False(gate.DeviceMemoryPointerExposed);
        Assert.False(gate.AllocationCallbackInvocationEnabled);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void CandidateListRecordsAllocatorInterfaceInfoBoundary()
    {
        JsonElement candidate = FindReadonlyCandidate("allocator-interface-info-design-004");

        Assert.Equal("design-gate-ready-with-safe-snapshot-alternative", candidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("copied allocator interface metadata snapshot", candidate.GetProperty("outputMode").GetString());
        AssertCandidateMethods(
            candidate,
            "IGpuAllocator::getInterfaceInfo",
            "IGpuAsyncAllocator::getInterfaceInfo",
            "IOutputAllocator::getInterfaceInfo");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/Callbacks/MemoryAllocation/TensorRtAllocatorInterfaceInfoDesignGate.cs");
        AssertEvidenceContains(candidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/AllocatorInterfaceInfoDesignGateTests.cs");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtAllocatorInterfaceInfoDesignGate.EvaluateKnownSurface");

        string ownership = candidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("allocator handles are not exposed", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("allocation callbacks remain deferred", ownership, StringComparison.OrdinalIgnoreCase);
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
