using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseEvidenceClassificationAuditTests
{
    [Fact]
    public void ClassificationAuditScriptCannotPublishOrPromoteProof()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-ReleaseEvidenceClassificationAudit.ps1"));

        foreach (string marker in new[]
        {
            "recordKind = \"release-evidence-classification-audit\"",
            "canPublishPublicly = $false",
            "canCloseReleaseIssue = $false",
            "isRuntimeExecutionProof = $false",
            "performsPublish = $false",
            "approvesPublicRelease = $false",
            "not runtime proof",
            "not permission to delete deferred records"
        })
        {
            Assert.Contains(marker, script, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", script, StringComparison.OrdinalIgnoreCase);
    }
}
