using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidateOwnerOneScreenExecutionPackTests
{
    [Fact]
    public void OneScreenExecutionPackKeepsReleaseBlockedUntilRealProofArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-owner-one-screen-execution-pack.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-candidate-owner-one-screen-execution-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("packState").GetString());
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
            "guidance only",
            "does not run models",
            "publish packages",
            "ProjectReference",
            "direct .nupkg",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OneScreenExecutionPackDefinesOrderedOwnerActionsForAllProofLanes()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-owner-one-screen-execution-pack.json");
        JsonElement root = document.RootElement;

        string[] expectedOrder =
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        };

        string[] ownerOrder = root.GetProperty("ownerExecutionOrder").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Equal(expectedOrder, ownerOrder);

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        for (int index = 0; index < expectedOrder.Length; index++)
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == expectedOrder[index]);
            Assert.Equal(index + 1, lane.GetProperty("ownerActionOrder").GetInt32());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("templateOrSchemaPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("commandToRun").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorCommand").GetString()));
            Assert.True(lane.GetProperty("requiredLogs").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredSha256Fields").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHostMetadata").GetArrayLength() >= 1);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("blockerDashboardLink").GetString()));
            Assert.True(lane.GetProperty("failureRepairHints").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 5);
        }
    }

    [Fact]
    public void OneScreenExecutionPackAlignsWithFinalFreezeAndReleaseIndexes()
    {
        using JsonDocument packDocument = ReadFinalReleaseJson("release-candidate-owner-one-screen-execution-pack.json");
        using JsonDocument freezeDocument = ReadFinalReleaseJson("real-external-execution-backfill-final-freeze.json");
        using JsonDocument ownerImportDocument = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        using JsonDocument closureIndexDocument = ReadFinalReleaseJson("release-evidence-closure-index.json");
        using JsonDocument closeBlockerDocument = ReadFinalReleaseJson("final-release-close-blocker-dashboard.json");
        using JsonDocument laneWorklistDocument = ReadFinalReleaseJson("release-close-proof-lane-worklist.json");

        JsonElement pack = packDocument.RootElement;
        string packText = pack.GetRawText();

        foreach (string surface in pack.GetProperty("linkedStatusSurfaces").EnumerateArray().Select(static item => item.GetString()!))
        {
            Assert.Contains(surface, pack.GetProperty("sourceArtifacts").GetRawText(), StringComparison.Ordinal);
        }

        Assert.Contains("release-candidate-owner-one-screen-execution-pack.json", freezeDocument.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.Contains("release-candidate-owner-one-screen-execution-pack.json", ownerImportDocument.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.Contains("release-candidate-owner-one-screen-execution-pack.json", closureIndexDocument.RootElement.GetRawText(), StringComparison.Ordinal);

        Assert.False(closeBlockerDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(closeBlockerDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(laneWorklistDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(laneWorklistDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());

        foreach (JsonElement freezeLane in freezeDocument.RootElement.GetProperty("lanes").EnumerateArray())
        {
            Assert.Contains(freezeLane.GetProperty("id").GetString()!, packText, StringComparison.Ordinal);
            Assert.Contains(freezeLane.GetProperty("expectedRecordPath").GetString()!, packText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void FinalAuditListsSurfacesClaimsAndRequiredBlockingMarkers()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-owner-one-screen-execution-pack.json");
        JsonElement audit = document.RootElement.GetProperty("finalAudit");

        Assert.Equal("blocked-owner-action-required", audit.GetProperty("auditState").GetString());

        string[] scannedSurfaces = audit.GetProperty("scannedSurfaces").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string surface in new[]
        {
            "README.md",
            "README.zh-CN.md",
            "docs/index.md",
            "docs/toc.yml",
            "artifacts/final-release/real-external-execution-backfill-final-freeze.json",
            "artifacts/final-release/owner-real-proof-import-master-pack.json",
            "artifacts/final-release/final-prepublish-quality-gate-dashboard.json",
            "artifacts/final-release/release-evidence-closure-index.json"
        })
        {
            Assert.Contains(surface, scannedSurfaces);
        }

        string[] forbiddenClaims = audit.GetProperty("forbiddenPositiveClaims").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string claim in new[]
        {
            "canPublishPublicly=true",
            "canCloseReleaseIssue=true",
            "currently publishable",
            "release issue can be closed",
            "template is proof",
            "matrix is proof",
            "report is proof",
            "blocked-by-cuda-driver is passed",
            "Skipped=True is passed",
            "local feed is package-consumer proof",
            "ProjectReference is package-consumer proof",
            "direct .nupkg is package-consumer proof"
        })
        {
            Assert.Contains(claim, forbiddenClaims);
        }

        string[] requiredBlockingMarkers = audit.GetProperty("requiredBlockingMarkers").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string marker in new[]
        {
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "blocked-owner-action-required",
            "template-only",
            "blocked-template-only",
            "not proof"
        })
        {
            Assert.Contains(marker, requiredBlockingMarkers);
        }
    }

    [Fact]
    public void FinalAuditScannedSurfacesKeepBlockedReleaseLanguage()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-owner-one-screen-execution-pack.json");
        JsonElement audit = document.RootElement.GetProperty("finalAudit");

        foreach (string relativePath in audit.GetProperty("scannedSurfaces").EnumerateArray().Select(static item => item.GetString()!))
        {
            string text = ReadText(relativePath.Split('/'));

            Assert.DoesNotContain("canPublishPublicly=true", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("blocked-by-cuda-driver is passed", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("Skipped=True is passed", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("local feed is package-consumer proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("ProjectReference is package-consumer proof", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("direct .nupkg is package-consumer proof", text, StringComparison.OrdinalIgnoreCase);

            bool hasBlockingMarker =
                text.Contains("canPublishPublicly=false", StringComparison.OrdinalIgnoreCase) ||
                text.Contains("canCloseReleaseIssue=false", StringComparison.OrdinalIgnoreCase) ||
                text.Contains("blocked-owner-action-required", StringComparison.OrdinalIgnoreCase) ||
                text.Contains("template-only", StringComparison.OrdinalIgnoreCase) ||
                text.Contains("not proof", StringComparison.OrdinalIgnoreCase);

            Assert.True(hasBlockingMarker, $"{relativePath} should retain at least one blocked/non-proof marker.");
        }
    }

    [Fact]
    public void OneScreenExecutionPackIsDocumentedAndCrossLinked()
    {
        string artifactMarkdown = ReadFinalReleaseText("release-candidate-owner-one-screen-execution-pack.md");
        string article = ReadText("docs", "articles", "zh-cn", "release-candidate-owner-one-screen-execution-pack.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Release Candidate Owner One-Screen Execution Pack",
            "blocked-owner-action-required",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "real-case-evidence-record.json",
            "package-consumer-runtime-proof-record.json",
            "post-publish-verification-record.json",
            "release-issue-close-record.json",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(marker, artifactMarkdown, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "发布候选 Owner 一屏执行包",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close",
            "local feed / ProjectReference / direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/release-candidate-owner-one-screen-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-candidate-owner-one-screen-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-candidate-owner-one-screen-execution-pack.md", docsToc, StringComparison.Ordinal);
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
