using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleProofBacklogTests
{
    [Fact]
    public void ExporterProducesStableFortyTwoArticleBacklogWithSixProofLanes()
    {
        RunPowerShell("Export-TechnicalArticleProofBacklog.ps1");
        string jsonPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "technical-article-proof-backlog.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "technical-article-proof-backlog.md");
        string firstJson = File.ReadAllText(jsonPath, Encoding.UTF8);
        string firstMarkdown = File.ReadAllText(markdownPath, Encoding.UTF8);
        string firstJsonHash = Hash(firstJson);
        string firstMarkdownHash = Hash(firstMarkdown);

        RunPowerShell("Export-TechnicalArticleProofBacklog.ps1");
        Assert.Equal(firstJsonHash, Hash(File.ReadAllText(jsonPath, Encoding.UTF8)));
        Assert.Equal(firstMarkdownHash, Hash(File.ReadAllText(markdownPath, Encoding.UTF8)));

        using JsonDocument document = JsonDocument.Parse(firstJson);
        JsonElement root = document.RootElement;
        Assert.Equal("technical-article-proof-backlog", root.GetProperty("recordKind").GetString());
        Assert.Equal(42, root.GetProperty("articleProofCount").GetInt32());
        Assert.Equal(89, root.GetProperty("proofRelationCount").GetInt32());
        Assert.Equal(6, root.GetProperty("laneCount").GetInt32());
        Assert.Equal(42, root.GetProperty("blockedArticleCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyArticleCount").GetInt32());
        Assert.Equal(103, root.GetProperty("contentCompleteCount").GetInt32());

        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] expectedLanes =
        [
            "callback-runtime",
            "linux-runner",
            "owner-authorization",
            "package-consumer-runtime",
            "post-publish-verification",
            "real-model-runtime"
        ];
        string[] actualLanes = root.GetProperty("laneSummaries")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Equal(expectedLanes, actualLanes);

        foreach (JsonElement article in root.GetProperty("articles").EnumerateArray())
        {
            Assert.True(article.GetProperty("contentComplete").GetBoolean());
            Assert.False(article.GetProperty("proofComplete").GetBoolean());
            Assert.NotEmpty(article.GetProperty("proofLanes").EnumerateArray());
            Assert.NotEmpty(article.GetProperty("validatorCommands").EnumerateArray());
            Assert.NotEmpty(article.GetProperty("firstCommands").EnumerateArray());
            JsonElement flags = article.GetProperty("promotionFlags");
            Assert.False(flags.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(flags.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(flags.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(flags.GetProperty("performsPublish").GetBoolean());
        }

        string markdown = File.ReadAllText(markdownPath, Encoding.UTF8);
        Assert.Contains("42", markdown, StringComparison.Ordinal);
        Assert.Contains("89", markdown, StringComparison.Ordinal);
        Assert.Contains("callback-runtime", markdown, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("canPublishPublicly=true", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("performsPublish=true", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BacklogReferencesAuthoritativeLedgerAndOwnerContractsWithoutPromotingProof()
    {
        RunPowerShell("Export-TechnicalArticleProofBacklog.ps1");
        string jsonPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publishing", "technical-article-proof-backlog.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath, Encoding.UTF8));
        JsonElement root = document.RootElement;

        Assert.Equal("docs/articles/zh-cn/publishing/technical-article-closure-ledger.json", root.GetProperty("sourceLedger").GetString());
        string[] sourcePaths = root.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetProperty("path").GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourcePaths);
        Assert.Contains("artifacts/final-release/real-owner-proof-convergence-dashboard-validation.json", sourcePaths);
        Assert.Contains("artifacts/final-release/owner-proof-execution-handoff.json", sourcePaths);
        Assert.Contains("artifacts/final-release/release-close-preflight.json", sourcePaths);

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("synthetic runtime", nonSubstitutes);

        foreach (JsonElement lane in root.GetProperty("laneSummaries").EnumerateArray())
        {
            Assert.Equal(0, lane.GetProperty("canPromoteProof").GetBoolean() ? 1 : 0);
            Assert.False(lane.GetProperty("performsPublish").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.NotEmpty(lane.GetProperty("requiredRealInputs").EnumerateArray());
            Assert.NotEmpty(lane.GetProperty("expectedArtifacts").EnumerateArray());
        }
    }

    private static string RunPowerShell(string scriptName)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                Arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{Path.Combine(RepositoryPaths.Root, "eng", scriptName)}\" -RepositoryRoot \"{RepositoryPaths.Root}\"",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"{scriptName} failed with exit code {process.ExitCode}. stdout={stdout} stderr={stderr}");
        return stdout;
    }

    private static string Hash(string value)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)));
    }
}
