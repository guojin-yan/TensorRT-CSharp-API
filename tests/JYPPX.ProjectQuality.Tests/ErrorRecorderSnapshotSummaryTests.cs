using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ErrorRecorderSnapshotSummaryTests
{
    [Fact]
    public void ErrorRecorderSnapshotExposesPointerFreeSummary()
    {
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtErrorRecorderSnapshot.cs");

        Assert.Contains("public TensorRtErrorRecorderSummary ToSummary()", snapshot);
        Assert.Contains("public sealed class TensorRtErrorRecorderSummary", snapshot);
        Assert.Contains("public int CopiedErrorRecordCount { get; }", snapshot);
        Assert.Contains("public bool CopiedRecordCountMatchesErrorCount", snapshot);
        Assert.Contains("public bool InterfaceInfoAvailable { get; }", snapshot);
        Assert.Contains("public string InterfaceName { get; }", snapshot);
        Assert.Contains("InterfaceInfo.Kind", snapshot);
        Assert.Contains("public int? FirstErrorCode { get; }", snapshot);
        Assert.Contains("public int FirstErrorDescriptionLength { get; }", snapshot);
        Assert.Contains("Records.Count > 0 ? Records[0] : null", snapshot);
        Assert.DoesNotContain("public IntPtr", snapshot);
        Assert.DoesNotContain("public nint", snapshot);
    }

    [Fact]
    public void CallbackSmokePrintsErrorRecorderSummaryMarker()
    {
        string program = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        Assert.Contains("TensorRtErrorRecorderSummary errorRecorderSummary = snapshot.ToSummary();", program);
        Assert.Contains("ErrorRecorderSummary={errorRecorderSummary.HasRecorder}/{errorRecorderSummary.ErrorCount}/{errorRecorderSummary.CopiedErrorRecordCount}/{errorRecorderSummary.InterfaceInfoAvailable}/{errorRecorderSummary.CopiedRecordCountMatchesErrorCount}", program);
        Assert.Contains("ErrorRecorderSummary=ToSummary;TensorRtErrorRecorderSummary;copied-record-count;pointer-free;not-runtime-proof", program);
    }

    [Fact]
    public void ExistingReadonlyDiagnosticsCandidateStillTargetsCopiedSnapshots()
    {
        string candidate = ReadSource("artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        string designGate = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtErrorRecorderDiagnosticsDesignGate.cs");

        Assert.Contains("TensorRtErrorRecorderSnapshot", candidate);
        Assert.Contains("direct IErrorRecorder ref-count ownership remains deferred by design", designGate);
        Assert.Contains("direct IErrorRecorder interface-info ownership remains deferred by design", designGate);
        Assert.Contains("RecorderPointerExposed", designGate);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
