using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofAutoSummaryOwnerBackfillCheckTests
{
    [Fact]
    public void SummaryPackKeepsReleaseBlockedAndNonProof()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-proof-auto-summary-owner-backfill-check.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-proof-auto-summary-owner-backfill-check", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("summaryState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        foreach (string marker in new[]
        {
            "blocker map only",
            "does not execute NuGet publish",
            "run models",
            "create real hashes",
            "existing JSON records",
            "local feed",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void SummaryPackAggregatesAllLanesWithBlockersAndValidators()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-proof-auto-summary-owner-backfill-check.json");
        JsonElement root = document.RootElement;

        string[] expectedOrder =
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        };

        Assert.Equal(expectedOrder, root.GetProperty("laneOrder").EnumerateArray().Select(static item => item.GetString()!).ToArray());

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        foreach (string laneId in expectedOrder)
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerEvidenceDirectory").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorOutputArchivePath").GetString()));
            Assert.True(lane.GetProperty("requiredLogFiles").GetArrayLength() >= 4);
            Assert.True(lane.GetProperty("requiredHashFields").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHostMetadata").GetArrayLength() >= 3);
            Assert.Contains("blocked", lane.GetProperty("currentProofState").GetString(), StringComparison.Ordinal);
            Assert.True(lane.GetProperty("remainingBlockers").GetArrayLength() >= 5);
            Assert.Contains("-FailOn", lane.GetProperty("strictValidatorCommand").GetString(), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SummaryPackDoesNotPromoteExistingJsonRecords()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-proof-auto-summary-owner-backfill-check.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime").GetProperty("currentFileExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime").GetProperty("currentFileExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "post-publish-verification").GetProperty("currentFileExists").GetBoolean());
        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "release-issue-close").GetProperty("currentFileExists").GetBoolean());

        foreach (JsonElement lane in lanes)
        {
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.Contains("blocked", lane.GetProperty("currentProofState").GetString(), StringComparison.Ordinal);
        }

        string rules = document.RootElement.GetProperty("summaryRules").GetRawText();
        foreach (string marker in new[]
        {
            "Summary pack itself is not proof",
            "Missing logs keep the lane blocked",
            "Missing hash fields keep the lane blocked",
            "Existing JSON records cannot promote without strict validator success"
        })
        {
            Assert.Contains(marker, rules, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SummaryPackIsLinkedFromReleaseClosureSurfaces()
    {
        string summaryPath = "artifacts/final-release/release-proof-auto-summary-owner-backfill-check.json";

        foreach (string fileName in new[]
        {
            "real-gpu-public-package-execution-backfill-pack.json",
            "prepublish-owner-evidence-input-landing-pack.json",
            "real-release-proof-backfill-final-close-gate.json",
            "owner-execution-result-import-prepublish-recheck.json",
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "release-evidence-closure-index.json"
        })
        {
            using JsonDocument linkedDocument = ReadFinalReleaseJson(fileName);
            Assert.Contains(summaryPath, linkedDocument.RootElement.GetRawText(), StringComparison.Ordinal);
            Assert.False(linkedDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(linkedDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void SummaryPackIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("release-proof-auto-summary-owner-backfill-check.md");
        string article = ReadText("docs", "articles", "zh-cn", "release-proof-auto-summary-owner-backfill-check.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Release Proof Auto Summary Owner Backfill Check",
            "blocked-owner-action-required",
            "Can publish publicly: `false`",
            "Can close release issue: `false`",
            "Summary pack itself is not proof"
        })
        {
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "Release Proof 自动汇总与 Owner 回填校验",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "Summary pack 本身不是 proof",
            "缺失日志",
            "strict validator"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/release-proof-auto-summary-owner-backfill-check.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-proof-auto-summary-owner-backfill-check.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-proof-auto-summary-owner-backfill-check.md", docsToc, StringComparison.Ordinal);
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
