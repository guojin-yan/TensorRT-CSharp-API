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
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1");
        string script = File.ReadAllText(scriptPath);

        Assert.Contains("Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)", script, StringComparison.Ordinal);
        Assert.DoesNotContain("$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
    }
}
