using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReadonlyDiagnosticsCandidateImplementationEvidenceTests
{
    [Fact]
    public void RemainingReadonlyDiagnosticsCandidatesAreLinkedToPointerFreeSnapshots()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        string candidateText = File.ReadAllText(candidatePath);
        using JsonDocument document = JsonDocument.Parse(candidateText);

        JsonElement groups = document.RootElement.GetProperty("groups");
        JsonElement errorRecorderCandidate = FindCandidate(groups, "error-recorder-snapshot-001");
        JsonElement dependencyCandidate = FindCandidate(groups, "dependency-readonly-diagnostics-001");

        AssertCandidateImplemented(errorRecorderCandidate);
        AssertCandidateImplemented(dependencyCandidate);

        AssertEvidenceContains(errorRecorderCandidate, "publicSurface", "TensorRtRuntime.TryGetErrorRecorderSnapshot");
        AssertEvidenceContains(errorRecorderCandidate, "publicSurface", "TensorRtErrorRecorderSnapshot");
        AssertEvidenceContains(errorRecorderCandidate, "publicSurface", "TensorRtExecutionContext.TryGetErrorRecorderSnapshot");
        AssertEvidenceContains(dependencyCandidate, "publicSurface", "TensorRtEnvironmentProbe.ProbeNativeDependencies");
        AssertEvidenceContains(dependencyCandidate, "publicSurface", "TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface");
        AssertEvidenceContains(dependencyCandidate, "publicSurface", "TensorRtDependencyProbeReport");

        string errorRecorderSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtErrorRecorderSnapshot.cs");
        string runtimeControls = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.Trt11Controls.cs");
        string refitterControls = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRefitter.Trt11Controls.cs");
        string engineControls = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEngine.Trt11BoundaryControls.cs");
        string contextControls = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtExecutionContext.Trt11BoundaryControls.cs");
        string runtimeSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntimeDiagnosticSnapshot.cs");
        string bridgeInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11BoundaryControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeSerializationRefit.cs");
        string dependencyProbe = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtEnvironmentProbe.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDependencyProbeReport.cs");
        string dependencyDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntimeDeserializationDependencyDiagnostics.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntimeDeserializationBoundaryPrecheck.cs");

        Assert.Contains("public sealed class TensorRtErrorRecorderSnapshot", errorRecorderSnapshot);
        Assert.Contains("public sealed class TensorRtErrorRecord", errorRecorderSnapshot);
        Assert.Contains("IReadOnlyList<TensorRtErrorRecord>", errorRecorderSnapshot);
        Assert.Contains("No native recorder pointer is exposed or retained", errorRecorderSnapshot);
        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", runtimeControls);
        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", refitterControls);
        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", engineControls);
        Assert.Contains("public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)", contextControls);
        Assert.Contains("public TensorRtRuntimeDiagnosticSnapshot GetDiagnosticSnapshot", runtimeControls);
        Assert.Contains("public sealed class TensorRtRuntimeDiagnosticSnapshot", runtimeSnapshot);
        Assert.Contains("GetRuntimeErrorRecorderSnapshot", bridgeInterop);
        Assert.Contains("GetRefitterErrorRecorderSnapshot", bridgeInterop);
        Assert.Contains("GetEngineErrorRecorderSnapshot", bridgeInterop);
        Assert.Contains("GetExecutionContextErrorRecorderSnapshot", bridgeInterop);

        Assert.Contains("public static TensorRtDependencyProbeReport ProbeNativeDependencies", dependencyProbe);
        Assert.Contains("public sealed class TensorRtDependencyProbeReport", dependencyProbe);
        Assert.Contains("public sealed class TensorRtNativeDependencyInfo", dependencyProbe);
        Assert.Contains("public static class TensorRtRuntimeDeserializationDependencyDiagnostics", dependencyDiagnostics);
        Assert.Contains("public readonly struct TensorRtRuntimeDeserializationDependencyDiagnosticsResult", dependencyDiagnostics);
        Assert.Contains("IsRuntimeExecutionProof => false", dependencyDiagnostics);
        Assert.Contains("IRuntime::loadRuntime remains deferred by design", dependencyDiagnostics);

        string publicText = string.Concat(
            errorRecorderSnapshot,
            runtimeControls,
            refitterControls,
            engineControls,
            contextControls,
            runtimeSnapshot,
            dependencyProbe,
            dependencyDiagnostics);

        Assert.DoesNotContain("public IntPtr", publicText);
        Assert.DoesNotContain("public nint", publicText);
        Assert.Contains("callback trampoline", candidateText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("borrowed pointer public exposure", candidateText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("callback trampoline implemented", candidateText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("borrowed pointer public exposure implemented", candidateText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EvidencePathsExistAndSmokeQualityCoverageNamesTheReadonlyDiagnostics()
    {
        string candidatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement groups = document.RootElement.GetProperty("groups");

        foreach (string candidateId in new[] { "error-recorder-snapshot-001", "dependency-readonly-diagnostics-001" })
        {
            JsonElement candidate = FindCandidate(groups, candidateId);
            JsonElement evidence = candidate.GetProperty("implementationEvidence");
            foreach (string bucket in new[] { "nativeSources", "managedSources", "smokeSources", "qualityTests" })
            {
                foreach (JsonElement item in evidence.GetProperty(bucket).EnumerateArray())
                {
                    string path = item.GetString() ?? string.Empty;
                    Assert.False(string.IsNullOrWhiteSpace(path));
                    Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, path)), $"{candidateId}:{bucket}:{path}");
                }
            }
        }

        string nativeCommon = ReadSource("native", "src", "tensorrt", "common", "error_recorder_boundary_controls.inc");
        string nativeTrt11 = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "boundary_controls.inc");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string errorRecorderTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "ErrorRecorderDiagnosticsDesignGateTests.cs");
        string dependencyTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "RuntimeDeserializationBoundaryPrecheckTests.cs");
        string packageReadinessTests = ReadSource("tests", "JYPPX.ProjectQuality.Tests", "RuntimePackageReadinessTests.cs");

        Assert.Contains("getNbErrors", nativeCommon + nativeTrt11);
        Assert.Contains("getErrorCode", nativeCommon + nativeTrt11);
        Assert.Contains("getErrorDesc", nativeCommon + nativeTrt11);
        Assert.Contains("hasOverflowed", nativeCommon + nativeTrt11);
        Assert.Contains("RuntimeDiagnosticSnapshot=", smoke);
        Assert.Contains("RuntimeDeserializationDependencyDiagnostics=", smoke);
        Assert.Contains("ErrorRecorderDiagnosticsDesignGate=", smoke);
        Assert.Contains("TensorRtEnvironmentProbe.ProbeNativeDependencies", smoke + dependencyTests);
        Assert.Contains("PublicGateSurfaceDoesNotExposeRawRecorderPointersOrOwnershipControls", errorRecorderTests);
        Assert.Contains("RuntimeDeserializationDependencyDiagnostics", dependencyTests);
        Assert.Contains("New-RuntimeDeserializationDependencyDiagnosticsEvidence", packageReadinessTests);
    }

    private static JsonElement FindCandidate(JsonElement groups, string candidateId)
    {
        foreach (JsonProperty group in groups.EnumerateObject())
        {
            foreach (JsonElement candidate in group.Value.EnumerateArray())
            {
                if (candidate.GetProperty("candidateId").GetString() == candidateId)
                {
                    return candidate;
                }
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }

    private static void AssertCandidateImplemented(JsonElement candidate)
    {
        Assert.Equal("implemented-with-pointer-free-wrapper", candidate.GetProperty("implementationStatus").GetString());
        Assert.True(candidate.TryGetProperty("implementationEvidence", out JsonElement evidence));
        Assert.True(evidence.GetProperty("managedSources").GetArrayLength() >= 4);
        Assert.True(evidence.GetProperty("smokeSources").GetArrayLength() >= 1);
        Assert.True(evidence.GetProperty("qualityTests").GetArrayLength() >= 1);
        Assert.True(evidence.GetProperty("publicSurface").GetArrayLength() >= 4);
        Assert.Contains("not expose", evidence.GetProperty("ownershipBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
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

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
