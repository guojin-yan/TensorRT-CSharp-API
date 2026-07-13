using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealReleaseProofBackfillFinalCloseGateTests
{
    [Fact]
    public void FinalCloseGateKeepsPublishAndCloseBlocked()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-release-proof-backfill-final-close-gate.json");
        JsonElement root = document.RootElement;

        Assert.Equal("real-release-proof-backfill-final-close-gate", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("gateState").GetString());
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
            "does not run models",
            "publish packages",
            "file existence",
            "matrix artifacts",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "local feed",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FinalCloseGateDefinesOrderedLanesAndRejectsFileExistencePromotion()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-release-proof-backfill-final-close-gate.json");
        JsonElement root = document.RootElement;

        string[] expectedOrder =
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        };

        Assert.Equal(
            expectedOrder,
            root.GetProperty("finalCloseOrder").EnumerateArray().Select(static item => item.GetString()!).ToArray());

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        for (int index = 0; index < expectedOrder.Length; index++)
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == expectedOrder[index]);

            Assert.Equal(index + 1, lane.GetProperty("closeOrder").GetInt32());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(lane.GetProperty("canCloseGatePass").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("currentBlockerState").GetString()));
            Assert.Contains("-FailOn", lane.GetProperty("requiredValidatorCommand").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("proofClassificationRequired").GetString()));
            Assert.True(lane.GetProperty("requiredLogs").GetArrayLength() >= 4);
            Assert.True(lane.GetProperty("requiredSha256Fields").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHostMetadata").GetArrayLength() >= 3);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("promotionCondition").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("closeGateCondition").GetString()));
            Assert.True(lane.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 7);
        }

        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime").GetProperty("currentRecordExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime").GetProperty("currentRecordExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "post-publish-verification").GetProperty("currentRecordExists").GetBoolean());
        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "release-issue-close").GetProperty("currentRecordExists").GetBoolean());
    }

    [Fact]
    public void FinalCloseGateCapturesPackageConsumerPostPublishAndCloseDependencies()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-release-proof-backfill-final-close-gate.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        JsonElement packageLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        string packageText = packageLane.GetRawText();
        foreach (string marker in new[] { "public package source", "no ProjectReference", "no local feed", "no direct .nupkg", "smokeStatus=passed", "nativeAssetsCopied=true" })
        {
            Assert.Contains(marker, packageText, StringComparison.Ordinal);
        }

        JsonElement postPublishLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "post-publish-verification");
        Assert.Contains("actual owner-authorized public publication", postPublishLane.GetRawText(), StringComparison.Ordinal);

        JsonElement closeLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "release-issue-close");
        Assert.Equal(
            new[] { "real-model-runtime", "package-consumer-runtime", "post-publish-verification" },
            closeLane.GetProperty("dependsOn").EnumerateArray().Select(static item => item.GetString()!).ToArray());
        Assert.Contains("must run last", closeLane.GetProperty("closeGateCondition").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalCloseGateIsCrossLinkedFromReleaseClosureSurfaces()
    {
        string gatePath = "artifacts/final-release/real-release-proof-backfill-final-close-gate.json";

        foreach (string fileName in new[]
        {
            "owner-execution-result-import-prepublish-recheck.json",
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "release-evidence-closure-index.json"
        })
        {
            using JsonDocument linkedDocument = ReadFinalReleaseJson(fileName);
            Assert.Contains(gatePath, linkedDocument.RootElement.GetRawText(), StringComparison.Ordinal);
            Assert.False(linkedDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(linkedDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void FinalCloseGateIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("real-release-proof-backfill-final-close-gate.md");
        string article = ReadText("docs", "articles", "zh-cn", "real-release-proof-backfill-final-close-gate.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Real Release Proof Backfill Final Close Gate",
            "blocked-owner-action-required",
            "Can publish publicly: `false`",
            "Can close release issue: `false`",
            "File existence alone is not proof",
            "release-issue-close-record.json"
        })
        {
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "真实发布 Proof 回填与最终关闭门",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "文件存在不等于 proof",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/real-release-proof-backfill-final-close-gate.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-release-proof-backfill-final-close-gate.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-release-proof-backfill-final-close-gate.md", docsToc, StringComparison.Ordinal);
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
