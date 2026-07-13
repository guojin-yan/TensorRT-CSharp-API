using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealExternalExecutionBackfillFinalFreezeTests
{
    [Fact]
    public void FinalFreezeKeepsReleaseBlockedUntilRealExternalExecutionArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-external-execution-backfill-final-freeze.json");
        JsonElement root = document.RootElement;

        Assert.Equal("real-external-execution-backfill-final-freeze", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("freezeState").GetString());
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
            "ProjectReference",
            "direct .nupkg",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }

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
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerImportSourceArtifact").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("currentValidationArtifact").GetString()));
            Assert.True(lane.GetProperty("requiredExternalExecutionEnvironment").GetArrayLength() >= 4);
            Assert.True(lane.GetProperty("missingOwnerInputs").GetArrayLength() >= 5);
            Assert.True(lane.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 5);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("finalFreezeBlockerReason").GetString()));
        }
    }

    [Fact]
    public void FinalFreezeAlignsWithOwnerImportDashboardAndReleaseClosureSurfaces()
    {
        using JsonDocument freezeDocument = ReadFinalReleaseJson("real-external-execution-backfill-final-freeze.json");
        using JsonDocument ownerImportDocument = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-prepublish-quality-gate-dashboard.json");
        using JsonDocument closureIndexDocument = ReadFinalReleaseJson("release-evidence-closure-index.json");
        using JsonDocument blockerDashboardDocument = ReadFinalReleaseJson("final-release-close-blocker-dashboard.json");
        using JsonDocument laneWorklistDocument = ReadFinalReleaseJson("release-close-proof-lane-worklist.json");

        JsonElement freeze = freezeDocument.RootElement;
        string freezeText = freeze.GetRawText();

        string[] linkedSurfaces = freeze.GetProperty("linkedFreezeSurfaces").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string required in new[]
        {
            "artifacts/final-release/owner-real-proof-import-master-pack.json",
            "artifacts/final-release/final-prepublish-quality-gate-dashboard.json",
            "artifacts/final-release/final-release-close-blocker-dashboard.json",
            "artifacts/final-release/release-close-proof-lane-worklist.json",
            "artifacts/final-release/release-evidence-closure-index.json"
        })
        {
            Assert.Contains(required, linkedSurfaces);
        }

        foreach (JsonElement ownerRecord in ownerImportDocument.RootElement.GetProperty("records").EnumerateArray())
        {
            Assert.Contains(ownerRecord.GetProperty("lane").GetString()!, freezeText, StringComparison.Ordinal);
            Assert.Contains(ownerRecord.GetProperty("expectedRecordPath").GetString()!, freezeText, StringComparison.Ordinal);
        }

        Assert.Contains("real-external-execution-backfill-final-freeze.json", ownerImportDocument.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.Contains("real-external-execution-backfill-final-freeze.json", dashboardDocument.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.Contains("real-external-execution-backfill-final-freeze.json", closureIndexDocument.RootElement.GetRawText(), StringComparison.Ordinal);

        Assert.False(blockerDashboardDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(blockerDashboardDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(laneWorklistDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(laneWorklistDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void FinalFreezeCapturesYoloVisionTaskDeltaAndPackageConsumerPreflightDelta()
    {
        using JsonDocument freezeDocument = ReadFinalReleaseJson("real-external-execution-backfill-final-freeze.json");
        using JsonDocument realCasePackDocument = ReadFinalReleaseJson("real-case-proof-execution-pack.json");
        using JsonDocument packageValidationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record-validation.json");

        JsonElement[] lanes = freezeDocument.RootElement.GetProperty("lanes").EnumerateArray().ToArray();
        JsonElement realModelLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime");
        JsonElement packageLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime");

        string[] tasks = realModelLane.GetProperty("taskDelta").EnumerateArray()
            .Select(static item => item.GetProperty("task").GetString()!)
            .ToArray();
        string[] packTasks = realCasePackDocument.RootElement.GetProperty("requiredTaskCoverage").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, tasks);
            Assert.Contains(task, packTasks);
        }

        foreach (JsonElement taskDelta in realModelLane.GetProperty("taskDelta").EnumerateArray())
        {
            string[] missing = taskDelta.GetProperty("missing").EnumerateArray()
                .Select(static item => item.GetString()!)
                .ToArray();

            foreach (string required in new[]
            {
                "modelSource",
                "modelLicense",
                "labels",
                "inputTensorOrImage",
                "engine",
                "outputArtifact",
                "stdoutLog",
                "stderrLog",
                "screenshot",
                "SHA256",
                "hostMetadata",
                "ownerReview"
            })
            {
                Assert.Contains(required, missing);
            }
        }

        string packageValidationText = packageValidationDocument.RootElement.GetRawText();
        foreach (string preflight in packageLane.GetProperty("finalPreflightDelta").EnumerateArray().Select(static item => item.GetString()!))
        {
            if (preflight is "public-package-source" or "smoke-log-exists" or "dependencyProbeStatus=passed")
            {
                continue;
            }

            Assert.Contains(preflight, packageValidationText, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "clean root outside repository",
            "public package source",
            "no ProjectReference",
            "no local feed",
            "no direct .nupkg",
            "smoke log exists",
            "dependencyProbeStatus=passed",
            "smokeStatus=passed",
            "nativeAssetsCopied=true"
        })
        {
            Assert.Contains(marker, packageLane.GetRawText(), StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FinalFreezeForbiddenSubstitutesStayAlignedWithClosureIndex()
    {
        using JsonDocument freezeDocument = ReadFinalReleaseJson("real-external-execution-backfill-final-freeze.json");
        using JsonDocument closureIndexDocument = ReadFinalReleaseJson("release-evidence-closure-index.json");

        string[] freezeForbidden = freezeDocument.RootElement.GetProperty("forbiddenSubstitutes").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] closureForbidden = closureIndexDocument.RootElement.GetProperty("forbiddenSubstitutes").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string forbidden in new[]
        {
            "YoloVision matrix",
            "TensorRtExec report",
            "OnnxToEngine report",
            "article",
            "roadmap",
            "template-only",
            "dry-run",
            "preflight-only",
            "sidecar-only",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver",
            "Skipped=True",
            "local feed",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(forbidden, freezeForbidden);
            Assert.Contains(forbidden, closureForbidden);
        }
    }

    [Fact]
    public void FinalFreezeIsDocumentedAndCrossLinked()
    {
        string artifactMarkdown = ReadFinalReleaseText("real-external-execution-backfill-final-freeze.md");
        string article = ReadText("docs", "articles", "zh-cn", "real-external-execution-backfill-final-freeze.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Real External Execution Backfill Final Freeze",
            "blocked-owner-action-required",
            "can publish publicly",
            "can close release issue",
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
            "真实外部执行回填与发布候选最终冻结",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close",
            "覆盖 YOLOv8n 的 `det/seg/pose/obb/cls/sem`",
            "不能替代完整 real-case proof",
            "no ProjectReference",
            "no direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/real-external-execution-backfill-final-freeze.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-external-execution-backfill-final-freeze.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-external-execution-backfill-final-freeze.md", docsToc, StringComparison.Ordinal);
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
