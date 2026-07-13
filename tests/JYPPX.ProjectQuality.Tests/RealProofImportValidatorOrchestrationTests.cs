using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealProofImportValidatorOrchestrationTests
{
    [Fact]
    public void OrchestrationPackKeepsReleaseBlockedAndReadOnly()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-proof-import-validator-orchestration.json");
        JsonElement root = document.RootElement;

        Assert.Equal("real-proof-import-validator-orchestration", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("orchestrationState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("read-only coordination", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("do not create real proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("execute NuGet publish", boundary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OrchestrationPackUsesExistingScriptEntrypoints()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-proof-import-validator-orchestration.json");
        JsonElement root = document.RootElement;

        foreach (string script in root.GetProperty("existingScriptEntrypoints").EnumerateArray().Select(static item => item.GetString()!))
        {
            Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, script)), $"{script} must exist.");
        }

        JsonElement summaryRunner = root.GetProperty("summaryRunner");
        Assert.Equal("eng/Test-ReleaseProofOwnerBackfillSummary.ps1", summaryRunner.GetProperty("path").GetString());
        Assert.Equal("read-only-summary", summaryRunner.GetProperty("mode").GetString());
        Assert.False(summaryRunner.GetProperty("canPromote").GetBoolean());
    }

    [Fact]
    public void OrchestrationPackDefinesAllLaneImportAndValidatorCommands()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-proof-import-validator-orchestration.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        foreach (string laneId in new[] { "real-model-runtime", "package-consumer-runtime", "post-publish-verification", "release-issue-close" })
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("requiredInputPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("importCommand").GetString()));
            Assert.Contains("-FailOn", lane.GetProperty("strictValidatorCommand").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("outputValidationSummaryPath").GetString()));
            Assert.Contains("blocked", lane.GetProperty("expectedMissingEvidenceState").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("proofClassificationExpected").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("promotionBlockerReason").GetString()));
        }
    }

    [Fact]
    public void SummaryRunnerWritesBlockedValidationSummary()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseProofOwnerBackfillSummary.ps1");
        Assert.True(File.Exists(scriptPath));

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("read-only", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPromote = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OrchestrationPackIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("real-proof-import-validator-orchestration.md");
        string article = ReadText("docs", "articles", "zh-cn", "real-proof-import-validator-orchestration.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        Assert.Contains("Real Proof Import Validator Orchestration", artifact, StringComparison.Ordinal);
        Assert.Contains("Can publish publicly: `false`", artifact, StringComparison.Ordinal);
        Assert.Contains("真实 Proof 导入脚本与严格 Validator 联动", article, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-proof-import-validator-orchestration.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-proof-import-validator-orchestration.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-proof-import-validator-orchestration.md", docsToc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return ReadText("artifacts", "final-release", fileName);
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
