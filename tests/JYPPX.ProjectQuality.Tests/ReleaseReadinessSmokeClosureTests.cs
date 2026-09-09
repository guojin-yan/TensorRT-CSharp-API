using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseReadinessSmokeClosureTests
{
    [Fact]
    public void YoloVisionMatrixAndManagedSmokeCoverEveryPromisedTask()
    {
        using JsonDocument matrix = ReadJson("applications", "YoloVision", "yolo-model-matrix.json");
        string matrixText = ReadText("applications", "YoloVision", "yolo-model-matrix.md");
        string tests = ReadText("tests", "JYPPX.ProjectQuality.Tests", "YoloVisionManagedPipelineTests.cs");

        string[] tasks = matrix.RootElement.GetProperty("tasks").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, tasks);
            Assert.Contains(task, matrixText, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "YoloTaskType.Detection",
            "YoloTaskType.Classification",
            "YoloTaskType.Segmentation",
            "YoloTaskType.OrientedBoundingBox",
            "YoloTaskType.Pose",
            "YoloTaskType.SemanticSegmentation",
            "DecodeOutput",
            "YoloCapabilityMatrix"
        })
        {
            Assert.Contains(marker, tests, StringComparison.Ordinal);
        }

        Assert.Contains("not real-model-runtime proof", matrix.RootElement.GetProperty("proofBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", matrixText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionAssetTemplatesRemainOwnerSuppliedAndNonPromotable()
    {
        string assetTemplate = ReadText("samples", "assets", "yolovision-assets.template.json");
        string sampleEvidenceTemplate = ReadText("artifacts", "user-acceptance", "sample-run-evidence-record.yolovision.template.json");
        string sidecarTemplate = ReadText("artifacts", "user-acceptance", "onnx-engine-build-evidence-sidecar.yolovision.template.json");
        string combined = assetTemplate + sampleEvidenceTemplate + sidecarTemplate;

        foreach (string marker in new[]
        {
            "owner-action-required",
            "template-only",
            "canPromoteRealModelRuntime",
            "false",
            "package-consumer runtime proof",
            "sampleRunLogSha256",
            "modelSha256",
            "labelsSha256",
            "inputAssetSha256"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("canPromoteRealModelRuntime\": true", combined, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("package-consumer-runtime is forbidden in sample run evidence records", combined, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not build-only, synthetic runtime, real-model runtime, or package-consumer runtime proof", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TrtexecParityMatrixIsBackedByParserAndApplicationSurfaces()
    {
        using JsonDocument parity = ReadJson("applications", "OnnxToEngine", "trtexec-parity-matrix.json");
        string parser = ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeParser.cs");
        string options = ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeOptions.cs");
        string deploymentOptions = string.Concat(
            ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeDeploymentOptions.cs"),
            ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeDeploymentOptions.Arguments.cs"),
            ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeDeploymentOptions.Diagnostics.cs"));
        string runtimeOptions = ReadText("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeRuntimeOptions.cs");
        string appOptions = ReadText("applications", "TensorRtExec", "Core", "TensorRtExecOptions.cs");
        string appService = ReadText("applications", "TensorRtExec", "Core", "TensorRtExecService.cs");
        string appCommand = ReadText("applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs");
        string combined = parser + options + deploymentOptions + runtimeOptions + appOptions + appService + appCommand;

        string[] requiredOptions =
        {
            "--onnx",
            "--saveEngine",
            "--loadEngine",
            "--minShapes",
            "--optShapes",
            "--maxShapes",
            "--shapes",
            "--fp16",
            "--int8",
            "--calib",
            "--workspace",
            "--memPoolSize",
            "--tacticSources",
            "--timingCacheFile",
            "--timingCache",
            "--profilingVerbosity",
            "--verbose",
            "--dryRun",
            "--sparsity",
            "--refit",
            "--device",
            "--useDLACore",
            "--allowGPUFallback"
        };

        foreach (string option in requiredOptions)
        {
            Assert.Contains(option, combined, StringComparison.Ordinal);
        }

        string parityText = parity.RootElement.GetRawText();
        foreach (string boundary in new[] { "parse-only", "build-only", "not runtime proof", "not real-model-runtime proof" })
        {
            Assert.Contains(boundary, parityText, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void TensorRtExecFeatureMatrixKeepsCliAndWinFormsOnSharedServicePath()
    {
        using JsonDocument featureMatrix = ReadJson("applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json");
        string command = ReadText("applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs");
        string form = ReadText("applications", "TensorRtExec", "WinForms", "MainForm.cs");
        string service = ReadText("applications", "TensorRtExec", "Core", "TensorRtExecService.cs");
        string options = ReadText("applications", "TensorRtExec", "Core", "TensorRtExecOptions.cs");

        Assert.Contains("new TensorRtExecService().Execute(options)", command, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecService", form, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecOptions", form, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecReport", service, StringComparison.Ordinal);
        Assert.Contains("TrtexecLikeParser.Parse", options, StringComparison.Ordinal);

        string matrixText = featureMatrix.RootElement.GetRawText();
        foreach (string status in new[] { "implemented", "wrapper-ready", "planned", "blocked by runtime proof" })
        {
            Assert.Contains(status, matrixText, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string forbidden in new[] { "canPublishPublicly=true", "canCloseReleaseIssue=true", "dotnet nuget push" })
        {
            Assert.DoesNotContain(forbidden, matrixText, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void PublicationArticleEntrypointsAreRealFilesWithBoundaries()
    {
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");
        string roadmap = ReadText("docs", "articles", "zh-cn", "publishing", "article-roadmap-30plus.json");

        (string FileName, string RoadmapMarker)[] entryArticles =
        {
            ("project-overview.md", "TensorRT CSharp API v4.0 项目总览"),
            ("onnx-to-engine-quickstart.md", "OnnxToEngine 快速入门"),
            ("tensorrtexec-cli-parameter-map.md", "TensorRtExec CLI"),
            ("yolovision-sample-overview.md", "YoloVision 总览"),
            ("plugin-inventory-readonly-api.md", "Plugin Registry Inventory")
        };

        foreach ((string articleFile, string roadmapMarker) in entryArticles)
        {
            string article = ReadText("docs", "articles", "zh-cn", articleFile);
            Assert.Contains(articleFile, docsIndex, StringComparison.Ordinal);
            Assert.Contains(articleFile, docsToc, StringComparison.Ordinal);
            Assert.Contains(roadmapMarker, roadmap, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("proof", article, StringComparison.OrdinalIgnoreCase);
            Assert.True(
                article.Contains("not", StringComparison.OrdinalIgnoreCase) ||
                article.Contains("不是", StringComparison.Ordinal) ||
                article.Contains("不能", StringComparison.Ordinal),
                articleFile);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
