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
        Assert.Contains("copied-diagnostics-boundary-present", script, StringComparison.Ordinal);
        Assert.Contains("parser-diagnostics-kind", script, StringComparison.Ordinal);
        Assert.Contains("parser-refitter-diagnostics-kind", script, StringComparison.Ordinal);
        Assert.Contains("copied-diagnostics-not-runtime-proof", script, StringComparison.Ordinal);
        Assert.Contains("parser-diagnostics-owner-action-present", script, StringComparison.Ordinal);
        Assert.Contains("copied-parser-diagnostics", script, StringComparison.Ordinal);
        Assert.Contains("copied-parser-refitter-diagnostics", script, StringComparison.Ordinal);
        Assert.Contains("OptionImplementationStatus", script, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", script, StringComparison.Ordinal);
        Assert.Contains("LoadedEngineDiagnostics", script, StringComparison.Ordinal);
        Assert.Contains("LayerInfoArtifact", script, StringComparison.Ordinal);
        Assert.Contains("layer-info-pointer-free", script, StringComparison.Ordinal);
        Assert.Contains("layer-info-export-file-match", script, StringComparison.Ordinal);
        Assert.Contains("BindingMetadata", script, StringComparison.Ordinal);
        Assert.Contains("binding-metadata-pointer-free", script, StringComparison.Ordinal);
        Assert.Contains("binding-metadata-not-runtime-proof", script, StringComparison.Ordinal);
        Assert.Contains("binding-metadata-not-release-proof", script, StringComparison.Ordinal);
        Assert.Contains("refit-lifecycle-consistent", script, StringComparison.Ordinal);
        Assert.Contains("refit-source-file-match", script, StringComparison.Ordinal);
        Assert.Contains("refit-persistence-consistent", script, StringComparison.Ordinal);
        Assert.Contains("refit-persistence-file-match", script, StringComparison.Ordinal);
        Assert.Contains("OriginalRefittedEngineDisposedBeforeReload", script, StringComparison.Ordinal);
        Assert.Contains("SerializationFlagsBefore", script, StringComparison.Ordinal);
        Assert.Contains("SerializationFlagsAfter", script, StringComparison.Ordinal);
        Assert.Contains("CapabilityProbe", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec and OnnxToEngine reports are build/report evidence", script, StringComparison.Ordinal);
        Assert.Contains("direct `.nupkg`", script, StringComparison.Ordinal);
        Assert.Contains("no-yolodet", script, StringComparison.Ordinal);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        Assert.Contains("Test-TensorRtExecReport.ps1", readme, StringComparison.Ordinal);
        Assert.Contains("ReportBoundary.ForbiddenSubstitutes", readme, StringComparison.Ordinal);
        Assert.Contains("ReportBoundary.CopiedDiagnosticsBoundary", readme, StringComparison.Ordinal);
        Assert.Contains("copied-parser-diagnostics", readme, StringComparison.Ordinal);
        Assert.Contains("copied-parser-refitter-diagnostics", readme, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", readme, StringComparison.Ordinal);
    }
}
