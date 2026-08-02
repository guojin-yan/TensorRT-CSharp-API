using System.Text.Json;
using System.Text.RegularExpressions;
using System.Diagnostics;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleRoadmapTests
{
    [Fact]
    public void TechnicalArticleRoadmapKeepsLargeNonDuplicateMatrix()
    {
        string roadmap = ReadRoadmap();
        int[] articleIds = ExtractArticleIds(roadmap);

        Assert.True(articleIds.Length >= 55, "The roadmap should remain a substantial article matrix, not a small placeholder list.");
        Assert.Equal(articleIds.Length, articleIds.Distinct().Count());
        Assert.Contains(81, articleIds);
        Assert.Contains(82, articleIds);
        Assert.Contains(83, articleIds);
        Assert.Contains(84, articleIds);
        Assert.Contains(85, articleIds);
        Assert.Contains(86, articleIds);
        Assert.Contains(87, articleIds);
        Assert.Contains(88, articleIds);
        Assert.Contains(89, articleIds);
        Assert.Contains(90, articleIds);
        Assert.Contains(91, articleIds);
        Assert.Contains(92, articleIds);
        Assert.Contains(93, articleIds);
        Assert.Contains(94, articleIds);
        Assert.Contains(95, articleIds);
        Assert.Contains(96, articleIds);
        Assert.Contains(97, articleIds);
        Assert.Contains(98, articleIds);
    }

    [Fact]
    public void TechnicalArticleRoadmapCoversPublicationSamplesAndProofBoundaries()
    {
        string roadmap = ReadRoadmap();
        string[] requiredMarkers =
        {
            "TensorRtExec",
            "OnnxToEngine",
            "YoloVision",
            "Classification",
            "Plugin Inventory",
            "C++",
            "CMake",
            "GitHub Release",
            "NuGet managed + bridge-only",
            "CUDA error 35",
            "blocked-by-cuda-driver",
            "build-only",
            "sample-run-evidence",
            "real-model-runtime",
            "package-consumer-runtime",
            "release proof record",
            "stale claim"
        };

        foreach (string marker in requiredMarkers)
        {
            Assert.Contains(marker, roadmap, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void TechnicalArticleRoadmapDefinesAuditableArticleQualityGateFields()
    {
        string roadmap = ReadRoadmap();

        Assert.Contains("### 文章质量门禁字段", roadmap, StringComparison.Ordinal);

        foreach (string requiredField in new[]
        {
            "article id",
            "标题",
            "类型",
            "对应 sample/application",
            "模型/资产",
            "模型获取方式",
            "license/hash 要求",
            "命令",
            "输出",
            "截图/图示需求",
            "proof boundary",
            "发布优先级",
            "完成状态"
        })
        {
            Assert.Contains(requiredField, roadmap, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string requiredBoundary in new[]
        {
            "samples/OnnxToEngine",
            "samples/YoloVision",
            "applications/TensorRtExec",
            "tensorrtsharp-source-build-cpp-guide.md",
            "source-build-windows-cpp-bridge.md",
            "source-build-cmake-presets-and-bindings.md",
            "nuget-github-dual-package-strategy.md",
            "build-only",
            "parse-only",
            "sidecar-only",
            "synthetic-input-runtime",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish verification",
            "model SHA256",
            "labels SHA256",
            "image SHA256",
            "日志 SHA256"
        })
        {
            Assert.Contains(requiredBoundary, roadmap, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", roadmap, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", roadmap, StringComparison.Ordinal);
    }

    [Fact]
    public void TechnicalArticlePublicationMatrixExportsAuditableReleaseFrozenPlan()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticlePublicationMatrix.ps1"));
        Assert.Contains("Technical article publication matrix written", output, StringComparison.Ordinal);

        JsonElement root = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "technical-article-publication-matrix.json"));

        Assert.Equal("technical-article-publication-matrix", root.GetProperty("recordKind").GetString());
        Assert.Equal("publication-planning-release-frozen", root.GetProperty("matrixState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(root.GetProperty("articleCount").GetInt32() >= 55);
        Assert.True(root.GetProperty("categoryCount").GetInt32() >= 8);
        Assert.True(root.GetProperty("yoloVisionArticleCount").GetInt32() >= 7);
        Assert.True(root.GetProperty("tensorRtExecArticleCount").GetInt32() >= 4);
        Assert.True(root.GetProperty("onnxToEngineArticleCount").GetInt32() >= 4);
        Assert.Contains("cannot substitute real owner authorization", root.GetProperty("releaseProofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("releaseCandidateClosureState").GetString());
        Assert.True(root.GetProperty("releaseCandidateClosureLaneCount").GetInt32() >= 5);
        Assert.Equal("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", root.GetProperty("yoloVisionScope").GetString());
        Assert.Contains("det/cls/seg/obb/pose/sem", root.GetProperty("yoloVisionTaskScope").GetString(), StringComparison.Ordinal);

        string[] forbiddenSubstitutes = root.GetProperty("releaseCandidateClosureForbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string forbiddenSubstitute in new[]
        {
            "local feed",
            "ProjectReference",
            "direct nupkg",
            "dry-run",
            "build-only",
            "sidecar-only",
            "dependency probe",
            "Skipped=True",
            "article matrix",
            "YoloVision matrix",
            "TensorRtExec report",
        })
        {
            Assert.Contains(forbiddenSubstitute, forbiddenSubstitutes);
        }

        string[] closureRequiredArtifacts = root.GetProperty("releaseCandidateClosureRequiredArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string requiredArtifact in new[]
        {
            "eng/Test-PackageConsumer.ps1",
            "pack/runtime-split/README.md",
            "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
            "samples/YoloVision/YoloVision.csproj",
            "artifacts/final-release/real-external-proof-record-import-validator.json",
        })
        {
            Assert.Contains(requiredArtifact, closureRequiredArtifacts);
        }

        JsonElement[] closureLanes = root.GetProperty("releaseCandidateClosureLanes").EnumerateArray().ToArray();
        Assert.Contains(closureLanes, static lane => lane.GetProperty("id").GetString() == "package-consumer-runtime-proof");
        Assert.Contains(closureLanes, static lane => lane.GetProperty("id").GetString() == "runtime-split-package-readiness");
        Assert.Contains(closureLanes, static lane => lane.GetProperty("id").GetString() == "publication-article-matrix");
        Assert.Contains(closureLanes, static lane => lane.GetProperty("id").GetString() == "sample-name-and-case-closure");
        Assert.Contains(closureLanes, static lane => lane.GetProperty("id").GetString() == "owner-real-release-proof");
        Assert.All(closureLanes, static lane => Assert.False(lane.GetProperty("canPromoteRuntimeProof").GetBoolean()));
        Assert.Contains(closureLanes, static lane => lane.GetProperty("nextOwnerAction").GetString()!.Contains("Owner", StringComparison.Ordinal));

        string[] requiredFields = root.GetProperty("requiredFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string requiredField in new[]
        {
            "articleId",
            "title",
            "category",
            "targetAudience",
            "coreScenario",
            "repoCodePaths",
            "sampleProject",
            "modelAssetsRequired",
            "commands",
            "screenshotsOrImagesNeeded",
            "proofDependencies",
            "publishChannelFit",
            "qualityNotes",
            "completionStatus",
        })
        {
            Assert.Contains(requiredField, requiredFields);
        }

        JsonElement[] articles = root.GetProperty("articles").EnumerateArray().ToArray();
        Assert.Equal(root.GetProperty("articleCount").GetInt32(), articles.Length);
        Assert.Equal(articles.Length, articles.Select(static article => article.GetProperty("articleId").GetInt32()).Distinct().Count());
        Assert.DoesNotContain(articles, static article => article.GetProperty("sampleProject").GetString()!.Contains("YoloDet", StringComparison.Ordinal));
        Assert.DoesNotContain(articles, static article => article.GetProperty("title").GetString()!.Contains("YoloDet", StringComparison.Ordinal));

        foreach (JsonElement article in articles)
        {
            Assert.True(article.GetProperty("articleId").GetInt32() > 0);
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("title").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("category").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("targetAudience").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("coreScenario").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("sampleProject").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("modelAssetsRequired").GetString()));
            Assert.True(article.GetProperty("commands").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("screenshotsOrImagesNeeded").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("proofDependencies").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("publishChannelFit").GetArrayLength() >= 1);
            Assert.True(article.GetProperty("qualityNotes").GetArrayLength() >= 1);
            Assert.False(string.IsNullOrWhiteSpace(article.GetProperty("completionStatus").GetString()));
        }

        JsonElement[] yoloArticles = articles
            .Where(static article =>
                article.GetProperty("title").GetString()!.Contains("Yolo", StringComparison.OrdinalIgnoreCase) ||
                article.GetProperty("coreScenario").GetString()!.Contains("YoloVision", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        Assert.True(yoloArticles.Length >= 7);
        Assert.Contains(yoloArticles, static article => article.GetProperty("qualityNotes").EnumerateArray().Any(static note => note.GetString()!.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", StringComparison.Ordinal)));
        Assert.Contains(yoloArticles, static article => article.GetProperty("commands").EnumerateArray().Any(static command => command.GetString()!.Contains(@".\samples\YoloVision", StringComparison.Ordinal)));

        JsonElement[] toolArticles = articles
            .Where(static article =>
                article.GetProperty("title").GetString()!.Contains("TensorRtExec", StringComparison.OrdinalIgnoreCase) ||
                article.GetProperty("title").GetString()!.Contains("OnnxToEngine", StringComparison.OrdinalIgnoreCase) ||
                article.GetProperty("coreScenario").GetString()!.Contains("TensorRtExec", StringComparison.OrdinalIgnoreCase) ||
                article.GetProperty("coreScenario").GetString()!.Contains("OnnxToEngine", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        Assert.True(toolArticles.Length >= 8);
        Assert.All(toolArticles, static article =>
            Assert.Contains(article.GetProperty("qualityNotes").EnumerateArray(), static note => note.GetString()!.Contains("build-only", StringComparison.OrdinalIgnoreCase)));

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "technical-article-publication-matrix.md"));
        Assert.Contains("Technical Article Publication Matrix", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
        Assert.Contains("Release Candidate Closure", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-real-proof-required", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("runtime-split-package-readiness", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-real-release-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", markdown, StringComparison.Ordinal);
        Assert.Contains("det/cls/seg/obb/pose/sem", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void SampleAndApplicationDocsDoNotPromoteBuildOnlyEvidenceToReleaseProof()
    {
        string samplesReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));
        string tensorRtExecReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));

        Assert.Contains("build-only", samplesReadme, StringComparison.Ordinal);
        Assert.Contains("not inference proof", samplesReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", samplesReadme, StringComparison.Ordinal);
        Assert.Contains("build-only/precheck output cannot be promoted to package-consumer-runtime proof", docsIndex, StringComparison.Ordinal);
        Assert.Contains("build-only", tensorRtExecReadme, StringComparison.Ordinal);
        Assert.Contains("不证明推理输出正确", tensorRtExecReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", tensorRtExecReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void SourceBuildAndDualPackageArticlesCoverCppBridgeNuGetAndGithubRoutes()
    {
        string sourceBuildGuide = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtsharp-source-build-cpp-guide.md"));
        string dualPackageGuide = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "nuget-github-dual-package-strategy.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string combined = sourceBuildGuide + dualPackageGuide + docsIndex + docsToc;

        foreach (string marker in new[]
        {
            "Visual Studio 2022",
            "CMake 3.27",
            "cmake --preset",
            "cmake --build",
            "eng\\Generate-Bindings.ps1",
            "eng\\Test-BindingGeneratorOutputs.ps1",
            "eng\\Test-BridgePackageConsumer.ps1",
            "JYPPX_TENSORRT_ROOT",
            "JYPPX_CUDA_ROOT",
            "JYPPX_CUDNN_ROOT",
            "GitHub Release",
            "NuGet-compatible source",
            "bridge-only",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "不是 release proof",
            "不是 clean public package runtime proof",
            "tensorrtsharp-source-build-cpp-guide.md"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void PublishReadinessArticlesAreLinkedAndGuardProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string[] articleFiles =
        {
            "sample-evidence-ladder.md",
            "onnxtoengine-and-tensorrtexec-boundary.md",
            "tool-report-to-release-proof-record.md",
            "stale-claim-prepublish-audit.md",
            "publish-final-mile-checklist.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string relativeHref = "articles/zh-cn/" + articleFile;
            string fullPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(fullPath);

            Assert.True(File.Exists(fullPath));
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
            Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
            Assert.Contains("build-only", article, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("release proof record", article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("sample-run-evidence", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "sample-evidence-ladder.md")), StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "onnxtoengine-and-tensorrtexec-boundary.md")), StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.json", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tool-report-to-release-proof-record.md")), StringComparison.Ordinal);
        Assert.Contains("ready-needs-manual-approval", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "stale-claim-prepublish-audit.md")), StringComparison.Ordinal);
        Assert.Contains("Post-publish verification", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publish-final-mile-checklist.md")), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseOwnerHandoffIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-owner-handoff.md");
        string ownerActionPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-action-required.md");
        string article = File.ReadAllText(articlePath);
        string ownerAction = File.ReadAllText(ownerActionPath);

        Assert.Contains("articles/zh-cn/release-owner-handoff.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-owner-handoff.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-owner-handoff.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-action-required.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("Owner action required: `artifacts/final-release/owner-action-required.md`", docsIndex, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "owner guidance",
            "release proof record",
            "external-runtime-proof-record.json",
            "post-publish-verification-record.json",
            "package-consumer-runtime proof",
            "real-model-runtime proof",
            "TrtexecAlignmentStatus=parse-only",
            "blocked-by-cuda-driver",
            "sidecar-only"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, ownerAction, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void RealProofBackfillPlaybooksAreLinkedAndGuardBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string[] articleFiles =
        {
            "package-consumer-runtime-proof-playbook.md",
            "post-publish-verification-proof-playbook.md",
            "real-model-evidence-backfill-playbook.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string relativeHref = "articles/zh-cn/" + articleFile;
            string fullPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(fullPath);

            Assert.True(File.Exists(fullPath));
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
            Assert.Contains("owner", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
            Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", article, StringComparison.Ordinal);
        }

        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-playbook.md")), StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "post-publish-verification-proof-playbook.md")), StringComparison.Ordinal);
        Assert.Contains("TrtexecAlignmentStatus=parse-only", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-model-evidence-backfill-playbook.md")), StringComparison.Ordinal);
    }

    [Fact]
    public void PublishableArticleBatchIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string[] articleFiles =
        {
            "tensorrtexec-option-layering-deep-dive.md",
            "yolovision-all-task-overview.md",
            "yolovision-detection-tutorial.md",
            "yolovision-segmentation-tutorial.md",
            "yolovision-pose-tutorial.md",
            "yolovision-obb-tutorial.md",
            "yolovision-classification-semantic-tutorial.md",
            "callback-allocator-safety-bridge-roadmap.md",
            "external-model-evidence-case-study.md",
            "project-release-story-and-boundaries.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string relativeHref = "articles/zh-cn/" + articleFile;
            string fullPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(fullPath);

            Assert.True(File.Exists(fullPath));
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
            Assert.Contains("build-only", article, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", article, StringComparison.Ordinal);
            Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
            Assert.Contains("owner", article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("TrtexecAlignmentStatus=parse-only", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-option-layering-deep-dive.md")), StringComparison.Ordinal);
        Assert.Contains("det、cls、seg、obb、pose、sem", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-all-task-overview.md")), StringComparison.Ordinal);
        Assert.Contains("real callback runtime proof", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "callback-allocator-safety-bridge-roadmap.md")), StringComparison.Ordinal);
        Assert.Contains("manifest/source 匹配不等于 100% 可用", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "project-release-story-and-boundaries.md")), StringComparison.Ordinal);
    }

    [Fact]
    public void ExternalModelEvidenceCaseStudyIsLongFormAuditableAndProofBounded()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "external-model-evidence-case-study.md"));

        Assert.True(article.Length >= 12_000, "The external-model evidence case study must remain a complete long-form tutorial.");

        foreach (string marker in new[]
        {
            "Acquisition Report",
            "Build Report / Sidecar",
            "Sample Run Evidence",
            "E:\\TensorRtSharpAssets\\cases\\<case-id>",
            "New-Item -ItemType Directory -Force",
            "Acquire-YoloXOfficialAssets.ps1 -Offline",
            "Acquire-YoloV10OfficialAssets.ps1 -Offline",
            "Test-SampleAssetManifest.ps1",
            "Test-OnnxEngineBuildEvidenceSidecar.ps1",
            "Test-YoloVisionOutputReport.ps1 -Strict",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "yolovision-real-asset-owner-backfill-pack.json",
            "packageConsumerRuntimeForbidden=true",
            "source-tree-real-model-runtime",
            "不是 package-consumer-runtime proof",
            "拒绝 C 盘"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        JsonElement yoloXManifest = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolox-official-assets.json"));
        JsonElement yoloXModel = yoloXManifest
            .GetProperty("assets")
            .EnumerateArray()
            .Single(static item => item.GetProperty("role").GetString() == "model");
        Assert.Contains(yoloXManifest.GetProperty("license").GetProperty("spdxId").GetString()!, article, StringComparison.Ordinal);
        Assert.Contains(yoloXManifest.GetProperty("upstreamTag").GetString()!, article, StringComparison.Ordinal);
        Assert.Contains(yoloXModel.GetProperty("expectedSha256").GetString()!, article, StringComparison.Ordinal);

        JsonElement yoloV10Manifest = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov10-official-assets.json"));
        JsonElement yoloV10Model = yoloV10Manifest
            .GetProperty("assets")
            .EnumerateArray()
            .Single(static item => item.GetProperty("role").GetString() == "model");
        Assert.Contains(yoloV10Manifest.GetProperty("license").GetProperty("spdxId").GetString()!, article, StringComparison.Ordinal);
        Assert.Contains(yoloV10Model.GetProperty("expectedSha256").GetString()!, article, StringComparison.Ordinal);

        JsonElement yoloV10Closure = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "yolov10-official-runtime-proof-closure.json"));
        JsonElement runtime = yoloV10Closure.GetProperty("runtime");
        Assert.Equal("source-tree-real-model-runtime", yoloV10Closure.GetProperty("proofClassification").GetString());
        Assert.Contains($"{runtime.GetProperty("predictionCount").GetInt32()} 个 predictions", article, StringComparison.Ordinal);
        Assert.Contains(runtime.GetProperty("topPrediction").GetString()!, article, StringComparison.Ordinal);
        Assert.Contains(runtime.GetProperty("topScore").GetDouble().ToString("0.########", System.Globalization.CultureInfo.InvariantCulture), article, StringComparison.Ordinal);

        foreach (string forbidden in new[]
        {
            "--model-path",
            "--engine-path",
            "--image-path",
            "--labels-path",
            "--output-path",
            "-RecordPath",
            "isPackageConsumerRuntimeProof=true",
            "canPublishPublicly=true",
            "performsPublish=true",
            "canCloseReleaseIssue=true"
        })
        {
            Assert.DoesNotContain(forbidden, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ProjectReleaseStoryMatchesCurrentCoverageMatricesAndFinalBlockers()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "project-release-story-and-boundaries.md"));
        Assert.True(article.Length >= 12_000, "The project release story must remain a complete long-form article.");

        string coverage = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "interface-coverage-summary.md"));
        Match manifestCount = Regex.Match(
            coverage,
            @"Manifest API count:\s*(?<count>\d+)",
            RegexOptions.CultureInvariant);
        Assert.True(manifestCount.Success, "The generated coverage summary must expose its manifest API count.");
        Assert.Contains(
            $"manifest API count：{manifestCount.Groups["count"].Value}",
            article,
            StringComparison.Ordinal);

        MatchCollection tensorRtSummaries = Regex.Matches(
            coverage,
            @"(?m)^- `TensorRT-(?<version>8\.6|10\.11|11\.0)[^`]*`: official interfaces scanned=(?<official>\d+), manifest matched=(?<matched>\d+), native source present=(?<source>\d+), implemented=(?<implemented>\d+), deferred-only=(?<deferred>\d+)\s*$",
            RegexOptions.CultureInvariant);
        foreach (string version in new[] { "8.6", "10.11", "11.0" })
        {
            Match[] versionSummaries = tensorRtSummaries
                .Cast<Match>()
                .Where(match => match.Groups["version"].Value == version)
                .ToArray();
            Assert.NotEmpty(versionSummaries);

            string[] distinctFacts = versionSummaries
                .Select(static match => string.Join(
                    '/',
                    match.Groups["official"].Value,
                    match.Groups["matched"].Value,
                    match.Groups["source"].Value,
                    match.Groups["implemented"].Value,
                    match.Groups["deferred"].Value))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            Assert.Single(distinctFacts);

            Match summary = versionSummaries[0];
            Assert.Contains(
                $"TensorRT {version}：{summary.Groups["official"].Value} scanned / " +
                $"{summary.Groups["matched"].Value} matched / " +
                $"{summary.Groups["source"].Value} source present / " +
                $"{summary.Groups["implemented"].Value} implemented / " +
                $"{summary.Groups["deferred"].Value} deferred-only",
                article,
                StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "manifest/source 匹配不等于 100% 可用",
            "TrtexecAlignmentStatus=parse-only",
            "60 行：55 supported、5 个 YOLOX",
            "GitHub Release 与 NuGet-compatible source 是两种获取通道",
            "managed core API",
            "项目自有 C++ bridge",
            "用户自行安装匹配的 TensorRT、CUDA、cuDNN",
            "New-Item -ItemType Directory -Force",
            "blocked-real-proof-required"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        JsonElement runtimePackages = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "release-candidate",
            "runtime-package-matrix.json"));
        int windowsPackageCount = runtimePackages.EnumerateArray().Count(static item => item.GetProperty("platform").GetString() == "windows");
        int linuxPackageCount = runtimePackages.EnumerateArray().Count(static item => item.GetProperty("platform").GetString() == "linux");
        Assert.Contains($"当前有 {runtimePackages.GetArrayLength()} 个 runtime keys", article, StringComparison.Ordinal);
        Assert.Contains($"{windowsPackageCount} 个 Windows", article, StringComparison.Ordinal);
        Assert.Contains($"{linuxPackageCount} 个 Linux", article, StringComparison.Ordinal);

        JsonElement fieldMap = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-gui-cli-field-map.json"));
        JsonElement gapList = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-release-candidate-gap-list.json"));
        Assert.Contains($"有 {fieldMap.GetProperty("fields").GetArrayLength()} 个字段", article, StringComparison.Ordinal);
        Assert.Contains($"有 {gapList.GetProperty("items").GetArrayLength()} 个 item", article, StringComparison.Ordinal);

        JsonElement publicationMatrix = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "technical-article-publication-matrix.json"));
        JsonElement selectedRoadmap = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "article-roadmap-30plus.json"));
        Assert.Contains($"有 {publicationMatrix.GetProperty("articleCount").GetInt32()} 篇文章记录", article, StringComparison.Ordinal);
        Assert.Contains($"有 {selectedRoadmap.GetProperty("articleCount").GetInt32()} 个精选 roadmap entries", article, StringComparison.Ordinal);

        JsonElement freeze = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-candidate-final-evidence-freeze.json"));
        Assert.Equal(5, freeze.GetProperty("blockerCount").GetInt32());
        Assert.False(freeze.GetProperty("performsPublish").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());
        foreach (JsonElement blocker in freeze.GetProperty("oneScreenReleaseHoldChecklist").EnumerateArray())
        {
            Assert.Contains(blocker.GetProperty("id").GetString()!, article, StringComparison.Ordinal);
        }

        foreach (string forbidden in new[]
        {
            "canPublishPublicly=true",
            "performsPublish=true",
            "canCloseReleaseIssue=true",
            "已完成公开发布",
            "已经发布到 NuGet",
            "已经发布到 GitHub Release"
        })
        {
            Assert.DoesNotContain(forbidden, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OwnerReleaseExecutionPackageArticleIsLinkedAndKeepsManualPublishBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-release-execution-package.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-release-execution-package.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/owner-release-execution-package.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-release-execution-package.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-release-execution-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-release-execution-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-OwnerReleaseExecutionPackage.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Owner release execution package: `artifacts/final-release/owner-release-execution-package.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "dotnet nuget push",
            "owner manually executes publish commands outside this script",
            "oneScreenReleaseHoldChecklist",
            "一屏 Release Hold 清单",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "TrtexecAlignmentStatus=parse-only",
            "ProjectReference",
            "sidecar-only"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("executedByThisScript = $false", script, StringComparison.Ordinal);
        Assert.Contains("owner-manual-command-placeholder", script, StringComparison.Ordinal);
    }

    [Fact]
    public void CompatibleHostProofBackfillPackageArticleIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "compatible-host-proof-backfill-package.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-backfill-package.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofBackfillPackage.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/compatible-host-proof-backfill-package.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/compatible-host-proof-backfill-package.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/compatible-host-proof-backfill-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/compatible-host-proof-backfill-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-CompatibleHostProofBackfillPackage.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Compatible host proof backfill package: `artifacts/final-release/compatible-host-proof-backfill-package.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "Owner Proof Final Backfill Tracks",
            "Required Owner Inputs",
            "Expected Artifacts",
            "Promotion Blockers",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json",
            "Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "mismatched log SHA256",
            "Windows handoff for Linux proof",
            "sidecar-only",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("ownerProofFinalBackfillTracks", script, StringComparison.Ordinal);
        Assert.Contains("requiredOwnerInputs", script, StringComparison.Ordinal);
        Assert.Contains("validatorCommands", script, StringComparison.Ordinal);
        Assert.Contains("expectedArtifacts", script, StringComparison.Ordinal);
        Assert.Contains("promotionBlockers", script, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", script, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json", script, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json", script, StringComparison.Ordinal);
        Assert.Contains("Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey", script, StringComparison.Ordinal);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", script, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFrontDoorNamesOwnerBackfillTracksAndNonSubstituteProofs()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string ownerChecklist = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-owner-action-checklist-final-hold.md"));
        string finalInspection = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-hold-final-inspection.md"));

        foreach (string content in new[] { readme, readmeZh, docsIndex, ownerChecklist, finalInspection })
        {
            Assert.Contains("canCloseReleaseIssue=false", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("ownerProofFinalBackfillTracks", content, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", content, StringComparison.Ordinal);
            Assert.Contains("linux-runner-proof", content, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", content, StringComparison.Ordinal);
            Assert.Contains("post-publish verification", content, StringComparison.Ordinal);
            Assert.Contains("local feed", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("ProjectReference", content, StringComparison.Ordinal);
            Assert.Contains("bridge-only", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Skipped=True", content, StringComparison.Ordinal);
            Assert.Contains("mismatched log SHA256", content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RealModelAndPackageProofInputPackageArticleIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-model-and-package-proof-input-package.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-model-and-package-proof-input-package.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-RealModelAndPackageProofInputPackage.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/real-model-and-package-proof-input-package.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-model-and-package-proof-input-package.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-model-and-package-proof-input-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-model-and-package-proof-input-package.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-RealModelAndPackageProofInputPackage.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Real model and package proof input package: `artifacts/final-release/real-model-and-package-proof-input-package.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"real-model-and-package-proof-input-package\"", script, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCloseGapDashboardArticleIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-close-gap-dashboard.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/release-close-gap-dashboard.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-gap-dashboard.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-gap-dashboard.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-close-gap-dashboard.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-ReleaseCloseGapDashboard.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Release close gap dashboard: `artifacts/final-release/release-close-gap-dashboard.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "recordKind=release-close-gap-dashboard",
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only",
            "real-model-and-package-proof-input-package",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("recordKind = \"release-close-gap-dashboard\"", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $canCloseReleaseIssue", script, StringComparison.Ordinal);
        Assert.Contains("input package", script, StringComparison.Ordinal);
    }

    [Fact]
    public void CompatibleHostProofExecutionPackArticleIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "compatible-host-proof-execution-pack.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-execution-pack.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofExecutionPack.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/compatible-host-proof-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/compatible-host-proof-execution-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/compatible-host-proof-execution-pack.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/compatible-host-proof-execution-pack.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-CompatibleHostProofExecutionPack.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Compatible host proof execution pack: `artifacts/final-release/compatible-host-proof-execution-pack.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "recordKind=compatible-host-proof-execution-pack",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("recordKind = \"compatible-host-proof-execution-pack\"", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", script, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof", script, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateFinalEvidenceFreezeArticleIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-candidate-final-evidence-freeze.md");
        string artifactPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.md");
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFinalEvidenceFreeze.ps1");

        Assert.True(File.Exists(articlePath));
        Assert.True(File.Exists(artifactPath));
        Assert.True(File.Exists(scriptPath));
        Assert.Contains("articles/zh-cn/release-candidate-final-evidence-freeze.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-candidate-final-evidence-freeze.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-candidate-final-evidence-freeze.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-candidate-final-evidence-freeze.md", roadmap, StringComparison.Ordinal);
        Assert.Contains("eng/Export-ReleaseCandidateFinalEvidenceFreeze.ps1", roadmap, StringComparison.Ordinal);
        Assert.Contains("Release candidate final evidence freeze: `artifacts/final-release/release-candidate-final-evidence-freeze.md`", docsIndex, StringComparison.Ordinal);

        string article = File.ReadAllText(articlePath);
        string artifact = File.ReadAllText(artifactPath);
        string script = File.ReadAllText(scriptPath);

        foreach (string marker in new[]
        {
            "recordKind=release-candidate-final-evidence-freeze",
            "freezeState=blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("recordKind = \"release-candidate-final-evidence-freeze\"", script, StringComparison.Ordinal);
        Assert.Contains("freezeState = \"blocked-real-proof-required\"", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFinalAuditArticleBatchIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string[] articleFiles =
        {
            "release-final-audit-map.md",
            "release-public-story-pack.md",
            "release-owner-proof-backlog.md",
            "release-proof-non-substitutes.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string relativeHref = "articles/zh-cn/" + articleFile;
            string fullPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(fullPath);

            Assert.True(File.Exists(fullPath));
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", article, StringComparison.Ordinal);
            Assert.Contains("post-publish verification", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
            Assert.Contains("build-only", article, StringComparison.Ordinal);
            Assert.Contains("sidecar-only", article, StringComparison.Ordinal);
            Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
            Assert.Contains("owner", article, StringComparison.OrdinalIgnoreCase);
        }

        string auditMap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-final-audit-map.md"));
        Assert.Contains("freezeState=blocked-real-proof-required", auditMap, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", auditMap, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", auditMap, StringComparison.Ordinal);
        Assert.Contains("Test-StaleReleaseClaims.ps1", auditMap, StringComparison.Ordinal);

        string publicStory = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-public-story-pack.md"));
        Assert.Contains("manifest/source 匹配和 deferred 边界已经可审计", publicStory, StringComparison.Ordinal);
        Assert.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", publicStory, StringComparison.Ordinal);
        Assert.Contains("det/cls/seg/obb/pose/sem", publicStory, StringComparison.Ordinal);

        string ownerBacklog = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-owner-proof-backlog.md"));
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", ownerBacklog, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof", ownerBacklog, StringComparison.Ordinal);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", ownerBacklog, StringComparison.Ordinal);

        string nonSubstitutes = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-proof-non-substitutes.md"));
        Assert.Contains("Release proof 必须满足四个条件", nonSubstitutes, StringComparison.Ordinal);
        Assert.Contains("把 `parse-only` 写成 implemented TensorRT 行为", auditMap, StringComparison.Ordinal);
        Assert.Contains("把 `sidecar-only` 写成 runtime proof", nonSubstitutes, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFrontpageArticleBatchIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string[] articleFiles =
        {
            "release-article-index-and-publishing-order.md",
            "release-readme-frontpage-checklist.md",
            "release-final-owner-action-sequence.md"
        };

        foreach (string articleFile in articleFiles)
        {
            string relativeHref = "articles/zh-cn/" + articleFile;
            string fullPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(fullPath);

            Assert.True(File.Exists(fullPath));
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", article, StringComparison.Ordinal);
            Assert.Contains("post-publish verification", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("blocked-by-cuda-driver", article, StringComparison.Ordinal);
            Assert.Contains("build-only", article, StringComparison.Ordinal);
            Assert.Contains("sidecar-only", article, StringComparison.Ordinal);
            Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
            Assert.Contains("YoloVision", article, StringComparison.Ordinal);
        }

        foreach (string content in new[] { readme, readmeZh })
        {
            Assert.Contains("blocked-real-proof-required", content, StringComparison.Ordinal);
            Assert.Contains("performsPublish=false", content, StringComparison.Ordinal);
            Assert.Contains("canPublishPublicly=false", content, StringComparison.Ordinal);
            Assert.Contains("canCloseReleaseIssue=false", content, StringComparison.Ordinal);
            Assert.Contains("release-issue-close-record-validation", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("blocked-template-only", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", content, StringComparison.Ordinal);
            Assert.Contains("release-final-audit-map.md", content, StringComparison.Ordinal);
            Assert.Contains("release-article-index-and-publishing-order.md", content, StringComparison.Ordinal);
            Assert.Contains("release-readme-frontpage-checklist.md", content, StringComparison.Ordinal);
            Assert.Contains("release-final-owner-action-sequence.md", content, StringComparison.Ordinal);
            Assert.Contains("TensorRtExec", content, StringComparison.Ordinal);
            Assert.Contains("YoloVision", content, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", content, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", content, StringComparison.Ordinal);
            Assert.Contains("post-publish verification", content, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("samples/YoloDet", content, StringComparison.Ordinal);
            Assert.DoesNotContain("YoloDet.csproj", content, StringComparison.Ordinal);
        }

        string publishingOrder = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-article-index-and-publishing-order.md"));
        Assert.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", publishingOrder, StringComparison.Ordinal);
        Assert.Contains("det、cls、seg、obb、pose、sem", publishingOrder, StringComparison.Ordinal);

        string ownerSequence = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-final-owner-action-sequence.md"));
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("Test-LinuxRunnerEvidenceRecord.ps1", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-template.json", ownerSequence, StringComparison.Ordinal);
        Assert.Contains("blocked-template-only", ownerSequence, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseFrontpageFinalAuditIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-frontpage-and-proof-boundary-final-audit.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains("release-frontpage-and-proof-boundary-final-audit.md", readme, StringComparison.Ordinal);
        Assert.Contains("release-frontpage-and-proof-boundary-final-audit.md", readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "build-only",
            "parse-only",
            "sidecar-only",
            "ProjectReference",
            "TensorRtExec",
            "YoloVision",
            "Test-StaleReleaseClaims.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateFinalCrossCheckIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-candidate-final-cross-check.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "build-only",
            "parse-only",
            "sidecar-only",
            "ProjectReference",
            "TensorRtExec",
            "YoloVision",
            "Test-StaleReleaseClaims.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "release-candidate-final-evidence-freeze.md",
            "release-frontpage-and-proof-boundary-final-audit.md",
            "release-proof-non-substitutes.md"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateArticleMatrixSummaryIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-candidate-article-matrix-summary.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "build-only",
            "parse-only",
            "sidecar-only",
            "ProjectReference",
            "TensorRtExec",
            "YoloVision",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
            "callback",
            "allocator",
            "debug listener",
            "Test-StaleReleaseClaims.ps1"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidatePublicationSummaryIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-candidate-publication-summary.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "build-only",
            "parse-only",
            "sidecar-only",
            "ProjectReference",
            "TensorRtExec",
            "YoloVision",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
            "Test-StaleReleaseClaims.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateFinalHoldOwnerWaitingIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-candidate-final-hold-owner-waiting.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=False",
            "canPublishPublicly=False",
            "canCloseReleaseIssue=False",
            "owner authorization",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification",
            "ProjectReference",
            "local feed",
            "build-only",
            "parse-only",
            "sidecar",
            "blocked-by-cuda-driver",
            "TensorRtExec",
            "YoloVision",
            "Test-StaleReleaseClaims.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseOwnerActionChecklistFinalHoldIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-owner-action-checklist-final-hold.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "PerformsPublish=False",
            "CanPublishPublicly=False",
            "CanCloseReleaseIssue=False",
            "owner authorization",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-SampleAssetManifest.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "ProjectReference",
            "local feed",
            "build-only",
            "parse-only",
            "sidecar-only",
            "blocked-by-cuda-driver",
            "TensorRtExec",
            "YoloVision"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFrontDoorKeepsFinalHoldState()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string frontDoor in new[] { readme, readmeZh })
        {
            Assert.Contains("blocked-real-proof-required", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("performsPublish=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("owner authorization", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("package-consumer-runtime", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("real-model-runtime", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("post-publish verification", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("release-owner-action-checklist-final-hold.md", frontDoor, StringComparison.Ordinal);
            Assert.Contains("release-candidate-final-hold-owner-waiting.md", frontDoor, StringComparison.Ordinal);
            Assert.Contains("release-hold-final-inspection.md", frontDoor, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReleaseHoldFinalInspectionIsLinkedAndKeepsProofBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string articleFile = "release-hold-final-inspection.md";
        string relativeHref = "articles/zh-cn/" + articleFile;
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
        Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        Assert.Contains(articleFile, readme, StringComparison.Ordinal);
        Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "release-candidate-final-evidence-freeze.json",
            "stale-release-claims-audit.json",
            "release-owner-action-checklist-final-hold.md",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification",
            "Test-StaleReleaseClaims.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "YoloVision",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
            "ProjectReference",
            "local feed",
            "build-only",
            "parse-only",
            "sidecar-only",
            "blocked-by-cuda-driver",
            "TensorRtExec"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFrontDoorLinksLatestFinalHoldArticles()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articleFile in new[]
        {
            "release-candidate-publication-summary.md",
            "release-candidate-final-hold-owner-waiting.md",
            "release-owner-action-checklist-final-hold.md",
            "release-hold-final-inspection.md"
        })
        {
            Assert.Contains(articleFile, readme, StringComparison.Ordinal);
            Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void StaleReleaseClaimsAuditHasNoFindings()
    {
        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "stale-release-claims-audit.json");

        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = audit.RootElement;

        Assert.Equal(0, root.GetProperty("findingCount").GetInt32());
        Assert.True(root.GetProperty("scannedFileCount").GetInt32() > 0);

        if (root.TryGetProperty("findings", out JsonElement findings))
        {
            Assert.Equal(0, findings.GetArrayLength());
        }
    }

    [Fact]
    public void ReleaseHoldLatestArticlesAreLinkedAcrossFrontDoorDocs()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = ReadRoadmap();
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articleFile in LatestFinalHoldArticleFiles())
        {
            string relativeHref = "articles/zh-cn/" + articleFile;

            Assert.Contains(articleFile, readme, StringComparison.Ordinal);
            Assert.Contains(articleFile, readmeZh, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsIndex, StringComparison.Ordinal);
            Assert.Contains(relativeHref, docsToc, StringComparison.Ordinal);
            Assert.Contains(relativeHref, roadmap, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TechnicalArticleRoadmapIdsAreContinuousThroughLatestReleaseHoldItems()
    {
        int[] articleIds = ExtractArticleIds(ReadRoadmap());
        int latestId = articleIds.Max();

        Assert.True(latestId >= 101);
        Assert.Contains(99, articleIds);
        Assert.Contains(100, articleIds);
        Assert.Contains(101, articleIds);
        Assert.Equal(Enumerable.Range(1, latestId), articleIds);
    }

    [Fact]
    public void ReleaseFreezeArtifactMatchesFrontDoorBlockedState()
    {
        string freezePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.json");
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(freezePath));
        JsonElement root = freeze.RootElement;

        Assert.Equal("blocked-real-proof-required", root.GetProperty("freezeState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(5, root.GetProperty("blockerCount").GetInt32());
        Assert.Equal(5, root.GetProperty("releaseClosePreflightFailedItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("staleReleaseClaimsFindingCount").GetInt32());

        JsonElement blockers = root.GetProperty("remainingBlockers");
        Assert.Equal(5, blockers.GetArrayLength());

        foreach (string gapId in new[]
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification"
        })
        {
            Assert.Contains(blockers.EnumerateArray(), blocker =>
                string.Equals(blocker.GetProperty("gapId").GetString(), gapId, StringComparison.Ordinal));
        }

        foreach (JsonElement blocker in blockers.EnumerateArray())
        {
            Assert.False(blocker.GetProperty("passed").GetBoolean());
            Assert.False(blocker.GetProperty("performsPublish").GetBoolean());
            Assert.False(blocker.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(blocker.GetProperty("canCloseReleaseIssue").GetBoolean());
        }

        foreach (string frontDoor in new[] { readme, readmeZh })
        {
            Assert.Contains("blocked-real-proof-required", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("performsPublish=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", frontDoor, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void LatestReleaseHoldArticlesDoNotUseRetiredYoloSampleName()
    {
        foreach (string articleFile in LatestFinalHoldArticleFiles())
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(articlePath);

            Assert.Contains("YoloVision", article, StringComparison.Ordinal);
            Assert.DoesNotContain("YoloDet", article, StringComparison.Ordinal);
            Assert.DoesNotContain("YoloDet.csproj", article, StringComparison.Ordinal);
            Assert.DoesNotContain("samples/YoloDet", article, StringComparison.Ordinal);
            Assert.DoesNotContain("samples\\YoloDet", article, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReleaseHoldDocsReferenceOwnerProofValidators()
    {
        foreach (string articleFile in LatestFinalHoldArticleFiles())
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(articlePath);

            foreach (string marker in new[]
            {
                "Test-StaleReleaseClaims.ps1",
                "Test-ExternalRuntimeProofRecord.ps1",
                "Test-PostPublishVerificationRecord.ps1",
                "Test-ReleaseIssueCloseRecord.ps1"
            })
            {
                Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
            }
        }

        string finalInspection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-hold-final-inspection.md"));

        foreach (string marker in new[]
        {
            "Test-SampleAssetManifest.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-ReleaseIssueCloseRecord.ps1",
            "release-issue-close-record-validation",
            "canCloseReleaseIssue=false"
        })
        {
            Assert.Contains(marker, finalInspection, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseCloseFacingDocsCarryIssueCloseRecordBoundary()
    {
        foreach (string articleFile in new[]
        {
            "release-close-preflight.md",
            "release-evidence-bundle.md",
            "owner-release-execution-package.md",
            "post-publish-verification-record.md",
            "release-proof-non-substitutes.md"
        })
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(articlePath);

            Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("release-issue-close-record", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("template-only", article, StringComparison.OrdinalIgnoreCase);
        }

        string closePreflight = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-close-preflight.md"));
        Assert.Contains("failedItemCount=9", closePreflight, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("releaseIssueCloseRecordValidationState=blocked-template-only", closePreflight, StringComparison.OrdinalIgnoreCase);

        string postPublish = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "post-publish-verification-record.md"));
        Assert.DoesNotContain("只有验证通过才允许 `canCloseReleaseIssue=true`", postPublish, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("不得单独写成 `canCloseReleaseIssue=true`", postPublish, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseFreezeRemainingBlockersCarryNonSubstituteBoundaries()
    {
        string freezePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.json");

        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(freezePath));
        JsonElement blockers = freeze.RootElement.GetProperty("remainingBlockers");

        Assert.Equal(5, blockers.GetArrayLength());

        foreach (JsonElement blocker in blockers.EnumerateArray())
        {
            Assert.False(blocker.GetProperty("passed").GetBoolean());
            Assert.False(blocker.GetProperty("performsPublish").GetBoolean());
            Assert.False(blocker.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(blocker.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("validator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("requiredRealEvidence").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("freezeBoundary").GetString()));

            string[] nonSubstitutes = blocker
                .GetProperty("nonSubstitutes")
                .EnumerateArray()
                .Select(static item => item.GetString() ?? string.Empty)
                .ToArray();

            foreach (string marker in new[]
            {
                "helper",
                "template",
                "draft",
                "runbook",
                "collection package",
                "input package",
                "local feed",
                "ProjectReference",
                "blocked-by-cuda-driver",
                "build-only",
                "parse-only",
                "sidecar-only"
            })
            {
                Assert.Contains(marker, nonSubstitutes);
            }
        }
    }

    [Fact]
    public void ReleaseHoldFrontDoorReferencesFreezeAndStaleAuditArtifacts()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string finalInspection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-hold-final-inspection.md"));

        foreach (string content in new[] { readme, readmeZh, finalInspection })
        {
            Assert.Contains("blocked-real-proof-required", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("release-candidate-final-evidence-freeze", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("stale-release-claims-audit", content, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("artifacts/final-release/release-candidate-final-evidence-freeze.json", finalInspection, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/stale-release-claims-audit.json", finalInspection, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseHoldArticlesKeepOwnerBlockerSetComplete()
    {
        foreach (string articleFile in LatestFinalHoldArticleFiles())
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(articlePath);

            foreach (string blocker in OwnerReleaseBlockerMarkers())
            {
                Assert.Contains(blocker, article, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    [Fact]
    public void ReleaseHoldDocsMatchFreezeBlockerProofClasses()
    {
        string freezePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.json");
        string ownerChecklist = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-owner-action-checklist-final-hold.md"));
        string finalInspection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-hold-final-inspection.md"));

        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(freezePath));
        JsonElement blockers = freeze.RootElement.GetProperty("remainingBlockers");

        var expectedProofClasses = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["owner-authorization"] = "owner-authorization",
            ["package-consumer-runtime"] = "package-consumer-runtime",
            ["linux-runner-proof"] = "linux-runner-proof",
            ["real-model-runtime"] = "real-model-runtime",
            ["post-publish-verification"] = "post-publish verification"
        };

        foreach (JsonElement blocker in blockers.EnumerateArray())
        {
            string gapId = blocker.GetProperty("gapId").GetString() ?? string.Empty;
            string proofClass = blocker.GetProperty("proofClass").GetString() ?? string.Empty;
            string validator = blocker.GetProperty("validator").GetString() ?? string.Empty;

            Assert.True(expectedProofClasses.TryGetValue(gapId, out string? expectedProofClass), $"Unexpected release blocker gapId: {gapId}");
            Assert.Equal(expectedProofClass, proofClass);
            Assert.False(blocker.GetProperty("passed").GetBoolean());

            foreach (string validatorName in ExtractValidatorNames(validator))
            {
                Assert.Contains(validatorName, ownerChecklist + finalInspection, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    [Fact]
    public void ReleaseFrontDoorDoesNotClaimPublicationCompletion()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string frontDoor in new[]
        {
            ExtractMarkdownSection(readme, "## Release Candidate Front Door", "## Current Verified Status"),
            ExtractMarkdownSection(readmeZh, "## 发布候选前台入口", "## 当前验证状态")
        })
        {
            Assert.Contains("blocked-real-proof-required", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", frontDoor, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", frontDoor, StringComparison.OrdinalIgnoreCase);

            foreach (string forbidden in new[]
            {
                "publicly released",
                "published to NuGet",
                "NuGet published",
                "post-publish verified",
                "post-publish verification completed",
                "release issue can be closed",
                "已经发布到 nuget.org",
                "post-publish 已验证",
                "release issue 可以关闭"
            })
            {
                Assert.DoesNotContain(forbidden, frontDoor, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    [Fact]
    public void ReleaseHoldDocsKeepArtifactAndValidatorChainComplete()
    {
        string ownerChecklist = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-owner-action-checklist-final-hold.md"));
        string finalInspection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-hold-final-inspection.md"));
        string combined = ownerChecklist + finalInspection;

        foreach (string artifact in new[]
        {
            "release-candidate-final-evidence-freeze.json",
            "release-candidate-final-evidence-freeze.md",
            "stale-release-claims-audit.json",
            "stale-release-claims-audit.md"
        })
        {
            Assert.Contains(artifact, finalInspection, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "owner authorization",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification",
            "Test-ReleaseOwnerApprovalInput.ps1",
            "Test-OwnerAuthorizedPublishCommandPlan.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-SampleAssetManifest.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseFinalHoldDoesNotHaveProofRecordsWithoutRealOwnerInputs()
    {
        JsonElement externalValidation = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-validation.json"));
        Assert.Contains(externalValidation.GetProperty("validationState").GetString(), new[]
        {
            "template-only",
            "draft-blocked-by-cuda-driver",
            "draft-rich-but-not-proof",
        });
        Assert.Contains(externalValidation.GetProperty("proofClassification").GetString(), new[]
        {
            "template-only",
            "dependency-probe-only",
        });
        Assert.False(externalValidation.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(externalValidation.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement externalTemplate = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-record-template.json"));
        Assert.True(externalTemplate.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", externalTemplate.GetProperty("proofClassification").GetString());
        Assert.False(externalTemplate.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(externalTemplate.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement externalDraft = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-record.draft.json"));
        Assert.True(externalDraft.GetProperty("templateOnly").GetBoolean());
        Assert.Contains(externalDraft.GetProperty("proofClassification").GetString(), new[]
        {
            "template-only",
            "dependency-probe-only",
        });
        Assert.False(externalDraft.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(externalDraft.GetProperty("isDependencyProbeOnly").GetBoolean());
        Assert.False(externalDraft.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains(externalDraft.GetProperty("currentRuntimeProofStatus").GetString(), new[]
        {
            "not-requested",
            "blocked-by-cuda-driver",
        });

        JsonElement postPublishValidation = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-validation.json"));
        Assert.Contains(postPublishValidation.GetProperty("validationState").GetString(), new[]
        {
            "template-only",
            "incomplete-post-publish-verification",
        });
        Assert.False(postPublishValidation.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishValidation.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement postPublishTemplate = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-record-template.json"));
        Assert.True(postPublishTemplate.GetProperty("templateOnly").GetBoolean());
        Assert.False(postPublishTemplate.GetProperty("performsPublish").GetBoolean());
        Assert.False(postPublishTemplate.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishTemplate.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseFinalHoldFrontDoorKeepsAllBlockersVisible()
    {
        string readmeFrontDoor = ExtractMarkdownSection(
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md")),
            "## Release Candidate Front Door",
            "## Current Verified Status");
        string readmeZhFrontDoor = ExtractMarkdownSection(
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md")),
            "## 发布候选前台入口",
            "## 当前验证状态");
        string finalInspection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-hold-final-inspection.md"));

        foreach (string content in new[] { readmeFrontDoor, readmeZhFrontDoor, finalInspection })
        {
            foreach (string blocker in OwnerReleaseBlockerMarkers())
            {
                Assert.Contains(blocker, content, StringComparison.OrdinalIgnoreCase);
            }

            Assert.Contains("blocked-real-proof-required", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("performsPublish=false", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canPublishPublicly=false", content, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", content, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseFinalHoldArtifactsStayBlockedUntilValidatorsPass()
    {
        JsonElement freeze = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-candidate-final-evidence-freeze.json"));
        Assert.Equal("blocked-real-proof-required", freeze.GetProperty("freezeState").GetString());
        Assert.Equal("blocked-real-proof-required", freeze.GetProperty("releaseClosePreflightState").GetString());
        Assert.Equal("blocked-real-proof-required", freeze.GetProperty("releaseCloseGapDashboardState").GetString());
        Assert.Equal("owner-action-required", freeze.GetProperty("compatibleHostProofExecutionPackState").GetString());
        Assert.Equal("owner-action-required", freeze.GetProperty("realModelAndPackageProofInputPackageState").GetString());
        Assert.Equal("blocked-real-proof-required", freeze.GetProperty("ownerReleaseExecutionPackageState").GetString());
        Assert.False(freeze.GetProperty("performsPublish").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement executionPack = ReadJsonRoot(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "compatible-host-proof-execution-pack.json"));
        Assert.Equal("owner-action-required", executionPack.GetProperty("packageState").GetString());
        Assert.Equal("blocked-real-proof-required", executionPack.GetProperty("releaseCloseGapDashboardState").GetString());
        Assert.Equal("owner-action-required", executionPack.GetProperty("compatibleHostProofBackfillPackageState").GetString());
        Assert.Equal("owner-action-required", executionPack.GetProperty("compatibleHostRuntimeProofCollectionBundleState").GetString());
        Assert.Equal("owner-action-required", executionPack.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.Equal("blocked-real-publication-required", executionPack.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.False(executionPack.GetProperty("performsPublish").GetBoolean());
        Assert.False(executionPack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(executionPack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(5, executionPack.GetProperty("blockerCount").GetInt32());

        foreach (JsonElement item in executionPack.GetProperty("executionItems").EnumerateArray())
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Equal("owner-action-required", item.GetProperty("ownerAction").GetString());
        }
    }

    private static string ReadRoadmap()
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));
    }

    private static JsonElement ReadJsonRoot(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string[] LatestFinalHoldArticleFiles()
    {
        return
        [
            "release-candidate-publication-summary.md",
            "release-candidate-final-hold-owner-waiting.md",
            "release-owner-action-checklist-final-hold.md",
            "release-hold-final-inspection.md"
        ];
    }

    private static string[] OwnerReleaseBlockerMarkers()
    {
        return
        [
            "owner authorization",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification"
        ];
    }

    private static string[] ExtractValidatorNames(string validator)
    {
        return Regex
            .Matches(validator, @"Test-[A-Za-z0-9]+\.ps1")
            .Select(static match => match.Value)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
    }

    private static string ExtractMarkdownSection(string markdown, string startHeading, string endHeading)
    {
        int start = markdown.IndexOf(startHeading, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Missing markdown section start: {startHeading}");

        int end = markdown.IndexOf(endHeading, start + startHeading.Length, StringComparison.Ordinal);
        Assert.True(end > start, $"Missing markdown section end: {endHeading}");

        return markdown[start..end];
    }

    private static int[] ExtractArticleIds(string roadmap)
    {
        return roadmap
            .Split(new[] { "\r\n", "\n" }, StringSplitOptions.None)
            .Select(static line => Regex.Match(line, @"^\|\s*(\d+)\s*\|"))
            .Where(static match => match.Success)
            .Select(static match => int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}
