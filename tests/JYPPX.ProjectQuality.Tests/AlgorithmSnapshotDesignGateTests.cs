using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class AlgorithmSnapshotDesignGateTests
{
    [Fact]
    public void AlgorithmSnapshotGateListsCallbackScopedCandidatesWithoutExposingPointers()
    {
        TensorRtAlgorithmSnapshotDesignGateResult gate =
            TensorRtAlgorithmSnapshotDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt10);

        Assert.Equal("callback-scoped copied algorithm timing, workspace, context, IO, and variant snapshot", gate.RequiredOutputMode);
        Assert.Contains("IAlgorithm", gate.CandidateInterfaces);
        Assert.Contains("IAlgorithmContext", gate.CandidateInterfaces);
        Assert.Contains("IAlgorithmIOInfo", gate.CandidateInterfaces);
        Assert.Contains("IAlgorithmVariant", gate.CandidateInterfaces);
        Assert.Contains("IAlgorithmSelector", gate.CandidateInterfaces);
        Assert.Contains("IAlgorithm::getTimingMSec", gate.CandidateMethods);
        Assert.Contains("IAlgorithm::getWorkspaceSize", gate.CandidateMethods);
        Assert.Contains("IAlgorithm::getAlgorithmVariant", gate.CandidateMethods);
        Assert.Contains("IAlgorithm::getAlgorithmIOInfoByIndex", gate.CandidateMethods);
        Assert.Contains("IAlgorithmContext::getName", gate.CandidateMethods);
        Assert.Contains("IAlgorithmContext::getNbInputs", gate.CandidateMethods);
        Assert.Contains("IAlgorithmContext::getNbOutputs", gate.CandidateMethods);
        Assert.Contains("IAlgorithmContext::getDimensions", gate.CandidateMethods);
        Assert.Contains("IAlgorithmIOInfo::getDataType", gate.CandidateMethods);
        Assert.Contains("IAlgorithmIOInfo::getStrides", gate.CandidateMethods);
        Assert.Contains("IAlgorithmIOInfo::getVectorizedDim", gate.CandidateMethods);
        Assert.Contains("IAlgorithmVariant::getImplementation", gate.CandidateMethods);
        Assert.Contains("IAlgorithmVariant::getTactic", gate.CandidateMethods);
        Assert.Contains("IAlgorithmSelector::getInterfaceInfo", gate.CandidateMethods);
        Assert.Equal(14, gate.CandidateMethodCount);
        Assert.Contains("callback-scoped algorithm result metadata", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=14", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("RequiredOutputMode=callback-scoped copied algorithm timing, workspace, context, IO, and variant snapshot", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.True(gate.DirectAlgorithmRowsDeferred);
        Assert.True(gate.DirectSelectorCallbackRowsDeferred);
        Assert.False(gate.AlgorithmPointerExposed);
        Assert.False(gate.AlgorithmContextPointerExposed);
        Assert.False(gate.AlgorithmIoInfoPointerExposed);
        Assert.False(gate.AlgorithmVariantPointerExposed);
        Assert.False(gate.AlgorithmSelectorCallbackTrampolineEnabled);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void AlgorithmSnapshotCandidateListRecordsDeferredRowsAndDesignGateBoundary()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement readonlyDiagnostics = document.RootElement.GetProperty("groups").GetProperty("readonlyDiagnostics");

        JsonElement candidate = FindCandidate(readonlyDiagnostics, "algorithm-snapshot-design-003");

        Assert.Equal("design-gate-ready-not-runtime-proof", candidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("callback-scoped copied algorithm timing, workspace, context, IO, and variant snapshot", candidate.GetProperty("outputMode").GetString());
        AssertCandidateMethods(
            candidate,
            "IAlgorithm::getTimingMSec",
            "IAlgorithm::getWorkspaceSize",
            "IAlgorithm::getAlgorithmVariant",
            "IAlgorithm::getAlgorithmIOInfoByIndex",
            "IAlgorithmContext::getName",
            "IAlgorithmContext::getNbInputs",
            "IAlgorithmContext::getNbOutputs",
            "IAlgorithmContext::getDimensions",
            "IAlgorithmIOInfo::getDataType",
            "IAlgorithmIOInfo::getStrides",
            "IAlgorithmIOInfo::getVectorizedDim",
            "IAlgorithmVariant::getImplementation",
            "IAlgorithmVariant::getTactic",
            "IAlgorithmSelector::getInterfaceInfo");

        AssertEvidenceContains(candidate, "nativeSources", "native/src/tensorrt/v8/modules/deferred/cross_version_tenth_batch_other_deferred.inc");
        AssertEvidenceContains(candidate, "nativeSources", "native/src/tensorrt/v10/modules/deferred/cross_version_other_deferred.inc");
        AssertEvidenceContains(candidate, "managedSources", "src/JYPPX.TensorRtSharp/TensorRtAlgorithmSnapshotDesignGate.cs");
        AssertEvidenceContains(candidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/AlgorithmSnapshotDesignGateTests.cs");
        AssertEvidenceContains(candidate, "publicSurface", "TensorRtAlgorithmSnapshotDesignGate.EvaluateKnownSurface");

        string ownership = candidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("borrowed IAlgorithm", ownership, StringComparison.Ordinal);
        Assert.Contains("not exposed", ownership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("callback", ownership, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonElement FindCandidate(JsonElement candidates, string candidateId)
    {
        foreach (JsonElement candidate in candidates.EnumerateArray())
        {
            if (candidate.GetProperty("candidateId").GetString() == candidateId)
            {
                return candidate;
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
