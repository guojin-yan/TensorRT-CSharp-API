using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerExecutionResultImportPrepublishRecheckTests
{
    [Fact]
    public void RecheckKeepsReleaseBlockedAndRejectsNonProofSubstitutes()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-execution-result-import-prepublish-recheck.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-execution-result-import-prepublish-recheck", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("recheckState").GetString());
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
            "template",
            "matrix",
            "build-only",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "local feeds",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void RecheckDefinesAllOwnerLanesWithRecordExistenceAndStrictValidators()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-execution-result-import-prepublish-recheck.json");
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
            root.GetProperty("ownerExecutionOrder").EnumerateArray().Select(static item => item.GetString()!).ToArray());

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        for (int index = 0; index < expectedOrder.Length; index++)
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == expectedOrder[index]);

            Assert.Equal(index + 1, lane.GetProperty("ownerActionOrder").GetInt32());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("templateOrPlaceholderState").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("templateOrSchemaPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("currentValidationArtifact").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("currentValidationState").GetString()));
            Assert.Contains("-FailOn", lane.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
            Assert.True(lane.GetProperty("requiredLogPaths").GetArrayLength() >= 4);
            Assert.True(lane.GetProperty("requiredSha256Fields").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHostMetadata").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("missingFields").GetArrayLength() >= 5);
            Assert.True(lane.GetProperty("repairHints").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 7);
        }

        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime").GetProperty("recordExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime").GetProperty("recordExists").GetBoolean());
        Assert.True(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "post-publish-verification").GetProperty("recordExists").GetBoolean());
        Assert.False(Assert.Single(lanes, item => item.GetProperty("id").GetString() == "release-issue-close").GetProperty("recordExists").GetBoolean());
    }

    [Fact]
    public void RecheckPromotionRulesPreventFileExistenceFromBecomingProof()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-execution-result-import-prepublish-recheck.json");
        JsonElement root = document.RootElement;

        string rules = root.GetProperty("promotionRules").GetRawText();
        foreach (string marker in new[]
        {
            "File existence alone is not proof",
            "Template or placeholder records remain blocked",
            "Local feed, ProjectReference, and direct .nupkg are forbidden",
            "Post-publish verification cannot promote before actual public publication",
            "Release issue close cannot promote before all real proof lanes pass"
        })
        {
            Assert.Contains(marker, rules, StringComparison.Ordinal);
        }

        string missing = root.GetProperty("aggregateMissingOwnerInputs").GetRawText();
        foreach (string marker in new[]
        {
            "real-case-evidence-record.json",
            "strict package-consumer runtime proof",
            "strict post-publish verification",
            "release-issue-close-record.json"
        })
        {
            Assert.Contains(marker, missing, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RecheckIsDocumentedAndCrossLinked()
    {
        string artifact = ReadFinalReleaseText("owner-execution-result-import-prepublish-recheck.md");
        string article = ReadText("docs", "articles", "zh-cn", "owner-execution-result-import-prepublish-recheck.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Owner Execution Result Import Prepublish Recheck",
            "blocked-owner-action-required",
            "Can publish publicly: `false`",
            "Can close release issue: `false`",
            "File existence alone is not proof",
            "package-consumer-runtime-proof-record.json",
            "post-publish-verification-record.json"
        })
        {
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "Owner 执行结果导入与发布前复验",
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

        Assert.Contains("articles/zh-cn/owner-execution-result-import-prepublish-recheck.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-execution-result-import-prepublish-recheck.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-execution-result-import-prepublish-recheck.md", docsToc, StringComparison.Ordinal);
    }

    [Fact]
    public void RecheckIsLinkedFromCurrentReleaseClosureSurfaces()
    {
        using JsonDocument recheckDocument = ReadFinalReleaseJson("owner-execution-result-import-prepublish-recheck.json");
        string recheckPath = "artifacts/final-release/owner-execution-result-import-prepublish-recheck.json";

        foreach (string fileName in new[]
        {
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "release-evidence-closure-index.json"
        })
        {
            using JsonDocument linkedDocument = ReadFinalReleaseJson(fileName);
            Assert.Contains(recheckPath, linkedDocument.RootElement.GetRawText(), StringComparison.Ordinal);
            Assert.False(linkedDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(linkedDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        }

        string linkedSurfaces = recheckDocument.RootElement.GetProperty("linkedStatusSurfaces").GetRawText();
        foreach (string surface in new[]
        {
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "final-prepublish-quality-gate-dashboard.json",
            "release-evidence-closure-index.json"
        })
        {
            Assert.Contains(surface, linkedSurfaces, StringComparison.Ordinal);
        }
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
