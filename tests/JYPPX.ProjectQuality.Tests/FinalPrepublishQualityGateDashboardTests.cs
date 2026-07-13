using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class FinalPrepublishQualityGateDashboardTests
{
    [Fact]
    public void DashboardKeepsFinalReleaseBlockedUntilRealOwnerProofArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("final-prepublish-quality-gate-dashboard.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-prepublish-quality-gate-dashboard", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("dashboardState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("does not run models", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("publish packages", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("blocked-by-cuda-driver", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Skipped", boundary, StringComparison.OrdinalIgnoreCase);

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        foreach (string laneId in new[]
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        })
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.True(lane.GetProperty("requiredBeforePromotion").GetArrayLength() >= 5);
            Assert.True(lane.GetProperty("missingOwnerInputs").GetArrayLength() >= 5);
            Assert.True(lane.GetProperty("cannotUse").GetArrayLength() >= 5);
            Assert.Contains("eng/Test-", lane.GetProperty("validator").GetString()!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DashboardAlignsWithClosureMapAndReleaseEvidenceIndex()
    {
        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-prepublish-quality-gate-dashboard.json");
        using JsonDocument closureMapDocument = ReadFinalReleaseJson("release-candidate-package-consumer-closure-map.json");
        using JsonDocument closureIndexDocument = ReadFinalReleaseJson("release-evidence-closure-index.json");

        JsonElement dashboard = dashboardDocument.RootElement;
        JsonElement closureMap = closureMapDocument.RootElement;
        JsonElement closureIndex = closureIndexDocument.RootElement;

        string[] dashboardLaneIds = dashboard.GetProperty("lanes").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        string[] closureMapLaneIds = closureMap.GetProperty("proofTracks").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        string[] closureGateIds = closureIndex.GetProperty("closureGates").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string laneId in dashboardLaneIds)
        {
            Assert.Contains(laneId, closureMapLaneIds);
            Assert.Contains(laneId, closureGateIds);
        }

        string[] dashboardForbidden = dashboard.GetProperty("forbiddenSubstitutes").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] closureIndexForbidden = closureIndex.GetProperty("forbiddenSubstitutes").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string closureMapText = closureMap.GetRawText();

        foreach (string forbidden in new[]
        {
            "YoloVision matrix",
            "TensorRtExec report",
            "OnnxToEngine report",
            "build-only",
            "dry-run",
            "sidecar-only",
            "template-only",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "dependency-probe-only",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "article",
            "roadmap"
        })
        {
            Assert.Contains(forbidden, dashboardForbidden);
            Assert.Contains(forbidden, closureIndexForbidden);
            Assert.Contains(forbidden, closureMapText, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void DashboardOwnerInputConvergenceMatchesSchemasAndValidationArtifacts()
    {
        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-prepublish-quality-gate-dashboard.json");
        using JsonDocument schemaDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.schema.json");
        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        using JsonDocument packageValidationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record-validation.json");
        using JsonDocument postPublishValidationDocument = ReadFinalReleaseJson("post-publish-verification-validation.json");
        using JsonDocument closeValidationDocument = ReadFinalReleaseJson("release-issue-close-record-validation.json");
        using JsonDocument realCasePackDocument = ReadFinalReleaseJson("real-case-proof-execution-pack.json");

        JsonElement convergence = dashboardDocument.RootElement.GetProperty("ownerInputConvergence");

        string[] packageFields = convergence.GetProperty("packageConsumerRequiredFields").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] schemaFields = schemaDocument.RootElement.GetProperty("fields").EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();
        JsonElement template = templateDocument.RootElement;

        foreach (string field in packageFields)
        {
            Assert.Contains(field, schemaFields);
            Assert.True(template.TryGetProperty(field, out _), $"Package consumer template is missing {field}.");
        }

        string[] realTasks = convergence.GetProperty("realCaseRequiredTasks").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] packTasks = realCasePackDocument.RootElement.GetProperty("requiredTaskCoverage").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, realTasks);
            Assert.Contains(task, packTasks);
        }

        string allValidationText = string.Join(
            '\n',
            packageValidationDocument.RootElement.GetRawText(),
            postPublishValidationDocument.RootElement.GetRawText(),
            closeValidationDocument.RootElement.GetRawText());

        foreach (string preflight in convergence.GetProperty("preflightChecks").EnumerateArray().Select(static item => item.GetString()!))
        {
            Assert.Contains(preflight, allValidationText, StringComparison.Ordinal);
        }

        Assert.Equal("template-only", packageValidationDocument.RootElement.GetProperty("validationState").GetString());
        Assert.Equal("incomplete-post-publish-verification", postPublishValidationDocument.RootElement.GetProperty("validationState").GetString());
        Assert.Equal("owner-action-required", postPublishValidationDocument.RootElement.GetProperty("postPublishProofClassification").GetString());
        Assert.False(postPublishValidationDocument.RootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.Equal("blocked-template-only", closeValidationDocument.RootElement.GetProperty("validationState").GetString());
        Assert.False(packageValidationDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(postPublishValidationDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(closeValidationDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void DashboardIsDocumentedAndCrossLinked()
    {
        string artifactMarkdown = ReadFinalReleaseText("final-prepublish-quality-gate-dashboard.md");
        string article = ReadText("docs", "articles", "zh-cn", "final-prepublish-quality-gate-dashboard.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Final Prepublish Quality Gate Dashboard",
            "blocked-owner-action-required",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "YoloVision matrix",
            "TensorRtExec report",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(marker, artifactMarkdown, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "发布前最终质量门 Dashboard",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close",
            "无 ProjectReference",
            "无 local feed",
            "无 direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/final-prepublish-quality-gate-dashboard.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/final-prepublish-quality-gate-dashboard.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-prepublish-quality-gate-dashboard.md", docsToc, StringComparison.Ordinal);
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
