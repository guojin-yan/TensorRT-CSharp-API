using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicApiDocumentationClosureTests
{
    [Fact]
    public void ClosureEvidenceRecordsBothZeroFindingTransitionsWithoutProofPromotion()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "public-api-documentation-closure.json");
        JsonElement root = document.RootElement;

        Assert.Equal("public-api-documentation-closure", root.GetProperty("recordKind").GetString());
        Assert.Equal(139, root.GetProperty("baseline").GetProperty("compilerCs1591WarningCount").GetInt32());
        Assert.Equal(24, root.GetProperty("baseline").GetProperty("bilingualFindingCount").GetInt32());
        Assert.Equal(0, root.GetProperty("closure").GetProperty("compilerCs1591WarningCount").GetInt32());
        Assert.Equal(0, root.GetProperty("closure").GetProperty("bilingualFindingCount").GetInt32());
        Assert.Equal(0, root.GetProperty("closure").GetProperty("backlogFindingCount").GetInt32());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string markdown = ReadText(
            "artifacts",
            "interface-coverage",
            "public-api-documentation-closure.md");
        Assert.Contains("139", markdown, StringComparison.Ordinal);
        Assert.Contains("24", markdown, StringComparison.Ordinal);
        Assert.Contains("0", markdown, StringComparison.Ordinal);
        Assert.Contains("不是 runtime execution proof", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ContinuousGateRunsCompilerAndBilingualAuditsWithoutSkippingBuild()
    {
        string workflow = ReadText(".github", "workflows", "release-quality-gate.yml");
        string releaseGate = ReadText("eng", "Test-ReleaseQualityGate.ps1");
        string bilingualAudit = ReadText("eng", "Test-PublicApiBilingualDocumentation.ps1");
        string compilerAudit = ReadText("eng", "Test-PublicApiDocumentation.ps1");

        Assert.Contains("Enforce public API documentation", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-PublicApiBilingualDocumentation.ps1", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Test-PublicApiBilingualDocumentation.ps1 -SkipBuild", workflow, StringComparison.Ordinal);
        Assert.Contains("workflow-public-api-documentation", releaseGate, StringComparison.Ordinal);
        Assert.Contains("Test-PublicApiDocumentation.ps1", bilingualAudit, StringComparison.Ordinal);
        Assert.Contains("JYPPXSuppressMissingXmlDocs=false", compilerAudit, StringComparison.Ordinal);
        Assert.Contains("warning CS1591", compilerAudit, StringComparison.Ordinal);
        Assert.Contains("if ($exitCode -ne 0)", compilerAudit, StringComparison.Ordinal);
        Assert.Contains("Public API documentation build failed", compilerAudit, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] pathParts) =>
        JsonDocument.Parse(ReadText(pathParts));

    private static string ReadText(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}
