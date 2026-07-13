using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofCommonIoTests
{
    [Fact]
    public void OwnerRealProofCommonUsesAtomicWritesAndRetryReads()
    {
        string commonPath = Path.Combine(RepositoryPaths.Root, "eng", "OwnerRealProofCommon.ps1");
        string common = File.ReadAllText(commonPath);

        Assert.Contains("function Write-Utf8FileAtomic", common, StringComparison.Ordinal);
        Assert.Contains("[System.IO.File]::Replace", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 10; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 8; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("ConvertFrom-Json", common, StringComparison.Ordinal);
        Assert.DoesNotContain("Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json", common, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnerRealPublishEvidenceIntakeDryRunPackUsesCommonAtomicWriterForJson()
    {
        foreach (string scriptName in new[]
        {
            "Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1",
            "Export-FinalReadonlyPublishAuditPack.ps1",
            "Test-FinalReadonlyPublishAuditPack.ps1",
            "Export-FinalOwnerOneScreenExecutionManual.ps1",
            "Test-FinalOwnerOneScreenExecutionManual.ps1",
            "Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1",
            "Export-PostPublishStrictCrossCheckPack.ps1",
            "Test-PostPublishStrictCrossCheckPack.ps1",
            "Export-ReleaseCloseStrictEvidenceClosure.ps1",
            "Test-ReleaseCloseStrictEvidenceClosure.ps1",
            "Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1",
            "Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1",
            "Export-PublicArticleReadinessMatrix.ps1",
            "Test-PublicArticleReadinessMatrix.ps1",
            "Export-FinalOwnerPublishEvidenceImportRunbook.ps1",
            "Test-FinalOwnerPublishEvidenceImportRunbook.ps1",
            "Export-PostPublishArticleProofGate.ps1",
            "Test-PostPublishArticleProofGate.ps1"
        })
        {
            string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
            string script = File.ReadAllText(scriptPath);

            Assert.Contains("Write-Utf8File -LiteralPath $jsonPath -InputObject", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
        }
    }
}
