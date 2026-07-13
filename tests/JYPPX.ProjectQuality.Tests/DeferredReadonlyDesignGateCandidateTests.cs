using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredReadonlyDesignGateCandidateTests
{
    [Fact]
    public void DimensionExpressionDesignGateListsDirectCandidatesWithoutExposingPointers()
    {
        TensorRtDimensionExpressionSnapshotDesignGateResult gate =
            TensorRtDimensionExpressionSnapshotDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("copied bool/int64 scalar snapshot tied to a proven owner object", gate.RequiredOutputMode);
        Assert.Contains("IDimensionExpr", gate.CandidateInterfaces);
        Assert.Contains("IExprBuilder", gate.CandidateInterfaces);
        Assert.Contains("IDimensionExpr::isConstant", gate.CandidateMethods);
        Assert.Contains("IDimensionExpr::isSizeTensor", gate.CandidateMethods);
        Assert.Contains("IDimensionExpr::getConstantValue", gate.CandidateMethods);
        Assert.Contains("IExprBuilder::constant", gate.CandidateMethods);
        Assert.Contains("IExprBuilder::operation", gate.CandidateMethods);
        Assert.Contains("IExprBuilder::declareSizeTensor", gate.CandidateMethods);
        Assert.Equal(6, gate.CandidateMethodCount);
        Assert.Contains("Identify a high-level owner", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=6", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("RequiredOutputMode=copied bool/int64 scalar snapshot tied to a proven owner object", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.False(gate.ExpressionPointerExposed);
        Assert.False(gate.ExprBuilderPointerExposed);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void ErrorRecorderDesignGateListsSafeSnapshotAlternativeForInterfaceInfo()
    {
        TensorRtErrorRecorderDiagnosticsDesignGateResult gate =
            TensorRtErrorRecorderDiagnosticsDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("owner-scoped copied diagnostics and interface metadata snapshot", gate.RequiredOutputMode);
        Assert.Contains("IErrorRecorder", gate.CandidateInterfaces);
        Assert.Contains("IErrorRecorder::getInterfaceInfo", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::getNbErrors", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::getErrorCode", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::getErrorDesc", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::hasOverflowed", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::incRefCount", gate.CandidateMethods);
        Assert.Contains("IErrorRecorder::decRefCount", gate.CandidateMethods);
        Assert.Equal(7, gate.CandidateMethodCount);
        Assert.Contains("owner-scoped snapshots", gate.NextSafeImplementationStep, StringComparison.Ordinal);
        Assert.Contains("CandidateMethodCount=7", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("RequiredOutputMode=owner-scoped copied diagnostics and interface metadata snapshot", gate.Diagnostic, StringComparison.Ordinal);
        Assert.True(gate.DesignGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.False(gate.RecorderPointerExposed);
        Assert.False(gate.RefCountPublicOwnershipControl);
        Assert.False(gate.InterfaceInfoPublicOwnershipControl);
        Assert.False(gate.CanPromoteRuntimeProof);
    }

    [Fact]
    public void MachineReadableCandidateListIncludesNextDesignGateBatch()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement readonlyDiagnostics = document.RootElement.GetProperty("groups").GetProperty("readonlyDiagnostics");

        JsonElement dimensionCandidate = FindCandidate(readonlyDiagnostics, "dimension-expression-snapshot-design-002");
        JsonElement recorderCandidate = FindCandidate(readonlyDiagnostics, "error-recorder-interface-info-design-002");

        Assert.Equal("design-gate-ready-not-runtime-proof", dimensionCandidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("design-gate-ready-with-safe-snapshot-alternative", recorderCandidate.GetProperty("implementationStatus").GetString());
        Assert.Equal("design gate for copied bool/int64 scalar snapshot", dimensionCandidate.GetProperty("outputMode").GetString());
        Assert.Equal("owner-scoped copied diagnostics and interface metadata snapshot", recorderCandidate.GetProperty("outputMode").GetString());

        AssertCandidateMethods(
            dimensionCandidate,
            "IDimensionExpr::isConstant",
            "IDimensionExpr::isSizeTensor",
            "IDimensionExpr::getConstantValue",
            "IExprBuilder::constant",
            "IExprBuilder::operation",
            "IExprBuilder::declareSizeTensor");

        AssertCandidateMethods(
            recorderCandidate,
            "IErrorRecorder::getInterfaceInfo",
            "IErrorRecorder::getNbErrors",
            "IErrorRecorder::getErrorCode",
            "IErrorRecorder::getErrorDesc",
            "IErrorRecorder::hasOverflowed",
            "IErrorRecorder::incRefCount",
            "IErrorRecorder::decRefCount");

        AssertEvidenceContains(dimensionCandidate, "managedSources", "src/JYPPX.TensorRtSharp/TensorRtDimensionExpressionSnapshotDesignGate.cs");
        AssertEvidenceContains(dimensionCandidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/DeferredReadonlyDesignGateCandidateTests.cs");
        AssertEvidenceContains(recorderCandidate, "managedSources", "src/JYPPX.TensorRtSharp/TensorRtErrorRecorderDiagnosticsDesignGate.cs");
        AssertEvidenceContains(recorderCandidate, "managedSources", "src/JYPPX.TensorRtSharp/TensorRtErrorRecorderSnapshot.cs");
        AssertEvidenceContains(recorderCandidate, "qualityTests", "tests/JYPPX.ProjectQuality.Tests/DeferredReadonlyDesignGateCandidateTests.cs");

        string dimensionOwnership = dimensionCandidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        string recorderOwnership = recorderCandidate.GetProperty("implementationEvidence").GetProperty("ownershipBoundary").GetString() ?? string.Empty;
        Assert.Contains("borrowed IDimensionExpr", dimensionOwnership, StringComparison.Ordinal);
        Assert.Contains("direct recorder pointers", recorderOwnership, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not exposed", recorderOwnership, StringComparison.OrdinalIgnoreCase);
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
