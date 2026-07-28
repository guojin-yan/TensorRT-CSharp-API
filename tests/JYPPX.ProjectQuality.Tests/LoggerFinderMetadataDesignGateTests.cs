using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class LoggerFinderMetadataDesignGateTests
{
    [Fact]
    public void LoggerFinderGateKeepsCallbackProviderPointerFree()
    {
        TensorRtLoggerFinderMetadataDesignGateResult gate =
            TensorRtLoggerFinderMetadataDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("copied logger finder interface metadata snapshot", gate.RequiredOutputMode);
        Assert.Contains("ILoggerFinder", gate.CandidateInterfaces);
        Assert.Contains("ILoggerFinder::getInterfaceInfo", gate.CandidateMethods);
        Assert.Equal(1, gate.CandidateMethodCount);
        Assert.Contains("logger callback lookup", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=1", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.DirectLoggerFinderRowsDeferred);
        Assert.False(gate.LoggerFinderPointerExposed);
        Assert.False(gate.LoggerCallbackPointerExposed);
        Assert.False(gate.LoggerFinderCallbackInvocationEnabled);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void CandidateListRecordsLoggerFinderBoundary()
    {
        JsonElement candidate = FindReadonlyCandidate("logger-finder-metadata-design-004");

        Assert.Equal("design-gate-ready-not-runtime-proof", candidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("copied logger finder interface metadata snapshot", candidate.GetProperty("outputMode").GetString());
        AssertCandidateMethods(candidate, "ILoggerFinder::getInterfaceInfo");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtLoggerFinderMetadataDesignGate.cs");
        AssertEvidenceContains(candidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/LoggerFinderMetadataDesignGateTests.cs");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtLoggerFinderMetadataDesignGate.EvaluateKnownSurface");

        string ownership = candidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("finder handle is not exposed", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("logger callback", ownership, StringComparison.OrdinalIgnoreCase);
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
