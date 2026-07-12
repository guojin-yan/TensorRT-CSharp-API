using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecReportValidatorTests
{
    [Fact]
    public void TensorRtExecReportValidatorScriptKeepsReportEvidenceNonProof()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Test-TensorRtExecReport.ps1");
        Assert.True(File.Exists(scriptPath), scriptPath);

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("tensor-rt-exec-report-validation", script, StringComparison.Ordinal);
        Assert.Contains("tensor-rt-exec-report-ready", script, StringComparison.Ordinal);
        Assert.Contains("ReportBoundary", script, StringComparison.Ordinal);
        Assert.Contains("report-boundary-not-runtime-proof", script, StringComparison.Ordinal);
        Assert.Contains("OptionImplementationStatus", script, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", script, StringComparison.Ordinal);
        Assert.Contains("LoadedEngineDiagnostics", script, StringComparison.Ordinal);
        Assert.Contains("CapabilityProbe", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec and OnnxToEngine reports are build/report evidence", script, StringComparison.Ordinal);
        Assert.Contains("direct `.nupkg`", script, StringComparison.Ordinal);
        Assert.Contains("no-yolodet", script, StringComparison.Ordinal);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        Assert.Contains("Test-TensorRtExecReport.ps1", readme, StringComparison.Ordinal);
        Assert.Contains("ReportBoundary.ForbiddenSubstitutes", readme, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", readme, StringComparison.Ordinal);
    }
}
