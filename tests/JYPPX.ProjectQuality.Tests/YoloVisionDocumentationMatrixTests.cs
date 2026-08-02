using Xunit;
using System.Text.Json;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionDocumentationMatrixTests
{
    [Fact]
    public void SegmentationTutorialBindsProbabilityMasksReportsValidatorsAndExplicitRemainingBoundary()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-segmentation-tutorial.md"));
        string detailedGuide = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-segmentation-mask-postprocess-guide.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string roadmap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));
        string schema = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "yolovision-output.schema.json"));
        string example = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "examples", "yolovision-output-seg.example.json"));
        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionOutputReport.ps1"));
        string articleCasePack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-article-case-pack.json"));
        string ownerBackfillPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.json"));
        string generatedOwnerBackfillPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.generated.json"));
        string candidateTemplate = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolov8-seg-candidate.template.json"));

        Assert.Contains("yolovision-segmentation-tutorial.md", readme, StringComparison.Ordinal);
        Assert.Contains("| 75 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "## Seg 与 Sem 不同",
            "## 当前实现边界",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg",
            "YoloDetection.SourceIndex",
            "[P,H,W]",
            "[1,P,H,W]",
            "ComposeLinearMask",
            "ComposeProbabilityMask",
            "稳定 sigmoid",
            "--mask-threshold 0.5",
            "--output-role-map boxes:det,proto:mask-prototypes",
            "--exportReport",
            "--preflight",
            "--output-json",
            "--visualization-svg",
            "maskTotalPixelCount",
            "maskValueKind",
            "maskPixelCountScope=prototype-grid-before-crop-resize",
            "data-mask-cell=\"true\"",
            "24x24",
            "--mask-spatial-transform",
            "--mask-coordinate-space model-input",
            "--mask-crop-to-box true",
            "[left, right) x [top, bottom)",
            "source-image-after-explicit-preprocess-inverse-and-optional-box-crop",
            "data-spatial-mask-cell=\"true\"",
            "YoloSegmentationSpatialTransform.cs",
            "Test-YoloVisionOutputReport.ps1",
            "Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "以下材料不得替代真实模型证明",
            "package-consumer-runtime",
            "## 发布前检查清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "maskTotalPixelCount",
            "maskValueKind",
            "maskPixelCountScope",
            "prototype-grid-before-crop-resize"
        })
        {
            Assert.Contains(marker, schema, StringComparison.Ordinal);
            Assert.Contains(marker, example, StringComparison.Ordinal);
            Assert.Contains(marker, validator, StringComparison.Ordinal);
        }

        foreach (string pack in new[] { articleCasePack, ownerBackfillPack, generatedOwnerBackfillPack, candidateTemplate })
        {
            Assert.Contains("--mask-threshold 0.5", pack, StringComparison.Ordinal);
            Assert.Contains("maskValueKind", pack, StringComparison.Ordinal);
            Assert.Contains("probability", pack, StringComparison.Ordinal);
            Assert.Contains("maskPixelCountScope", pack, StringComparison.Ordinal);
            Assert.Contains("prototype-grid-before-crop-resize", pack, StringComparison.Ordinal);
            Assert.Contains("maskSpatialTransform", pack, StringComparison.Ordinal);
            Assert.Contains("explicit-preprocess-inverse", pack, StringComparison.Ordinal);
            Assert.Contains("maskCoordinateSpace", pack, StringComparison.Ordinal);
            Assert.Contains("model-input-pixels", pack, StringComparison.Ordinal);
            Assert.Contains("maskCropToDetection", pack, StringComparison.Ordinal);
            Assert.Contains("source-image-after-explicit-preprocess-inverse-and-optional-box-crop", pack, StringComparison.Ordinal);
            Assert.Contains("owner must validate exporter-specific mask alignment", pack, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloVisionRuntimePipeline.cs", detailedGuide, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloVisionSegmentationDecoder.cs", detailedGuide, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloVisionMaskComposer.cs", detailedGuide, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloVisionNms.cs", detailedGuide, StringComparison.Ordinal);
        Assert.Contains("YoloSampleRunner.cs", detailedGuide, StringComparison.Ordinal);
        Assert.Contains("YoloMaskComposer.cs", detailedGuide, StringComparison.Ordinal);
    }

    [Fact]
    public void AllTaskOverviewAndDetectionTutorialAreLongFormExecutableAndContractAligned()
    {
        string overview = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-all-task-overview.md"));
        string detection = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-detection-tutorial.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string roadmap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));
        string candidate = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolov8-det-candidate.template.json"));
        string articlePack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-article-case-pack.json"));
        string ownerPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.json"));
        string generatedOwnerPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.generated.json"));
        string ownerPackExporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerBackfillPack.ps1"));

        Assert.True(overview.Length >= 12000, $"All-task overview is too short: {overview.Length} characters.");
        Assert.True(detection.Length >= 12000, $"Detection tutorial is too short: {detection.Length} characters.");
        Assert.Contains("yolovision-all-task-overview.md", readme, StringComparison.Ordinal);
        Assert.Contains("yolovision-detection-tutorial.md", readme, StringComparison.Ordinal);
        Assert.Contains("| 73 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("| 74 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "## 最重要的概念：三层能力与证据",
            "YoloCapabilityMatrix.cs",
            "共生成 60 行",
            "其中 55 行",
            "5 行 unsupported",
            "yolo-model-matrix.json",
            "yolovision-task-output-contract.json",
            "YOLOv5",
            "YOLOv6",
            "YOLOv7",
            "YOLOv8",
            "YOLOv9",
            "YOLOv10",
            "YOLOv11",
            "YOLOv26",
            "YOLOX",
            "custom",
            "det",
            "cls",
            "seg",
            "obb",
            "pose",
            "sem",
            "E:\\TensorRtSharpAssets\\cases",
            "--list-capabilities --json",
            "--exportReport",
            "--image",
            "--preprocessed-output",
            "以下材料不得替代真实模型证明",
            "package-consumer-runtime",
            "## 发布前检查清单"
        })
        {
            Assert.Contains(marker, overview, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "## 路径一：Generic Raw Head",
            "## 路径二：YOLOv10 End-to-End",
            "## 路径三：YOLOX",
            "[1,C,N]",
            "[1,N,C]",
            "[1,N,6]",
            "score = objectness * max(classScores)",
            "Fail-Closed 数值检查",
            "SourceIndex",
            "--layout end2end",
            "no second NMS",
            "[1,8400,85]",
            "strides：8、16、32",
            "不会自动把模型输出框逆 letterbox 到 source-image 坐标",
            "coordinateSpace=model-input-pixels|normalized|source-image-pixels",
            "--image",
            "--preprocessed-output",
            "--output-json",
            "--visualization-svg",
            "Test-YoloVisionOutputReport.ps1 -Strict",
            "Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "以下材料不得替代真实模型证明",
            "package-consumer-runtime",
            "## 发布前检查清单"
        })
        {
            Assert.Contains(marker, detection, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string invalidOption in new[]
        {
            "--exportProfile",
            "--engine ",
            "--output-layout",
            "--confidence-threshold",
            ".jpg"
        })
        {
            Assert.DoesNotContain(invalidOption, overview, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(invalidOption, detection, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string pack in new[] { candidate, articlePack, ownerPack, generatedOwnerPack })
        {
            Assert.Contains(".ppm", pack, StringComparison.Ordinal);
            Assert.Contains("--preprocessed-output", pack, StringComparison.Ordinal);
            Assert.Contains("--output-json", pack, StringComparison.Ordinal);
            Assert.Contains("--visualization-svg", pack, StringComparison.Ordinal);
            Assert.Contains("coordinateSpace", pack, StringComparison.Ordinal);
            Assert.Contains("letterboxContract", pack, StringComparison.Ordinal);
            Assert.Contains("sourceImageInversePolicy", pack, StringComparison.Ordinal);
            Assert.Contains("not-automatic-owner-transform-required", pack, StringComparison.Ordinal);
        }

        Assert.Contains("yolov8-det-test.ppm", candidate, StringComparison.Ordinal);
        foreach (string pack in new[] { articlePack, ownerPack, generatedOwnerPack })
        {
            Assert.Contains("yolov8n-det.ppm", pack, StringComparison.Ordinal);
        }

        Assert.Contains("model-input-pixels-or-owner-confirmed", ownerPackExporter, StringComparison.Ordinal);
        Assert.Contains("not-automatic-owner-transform-required", ownerPackExporter, StringComparison.Ordinal);
    }

    [Fact]
    public void PoseTutorialIsLongFormAndMatchesManagedKeypointOwnershipBoundary()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-pose-tutorial.md"));
        string roadmap = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "technical-article-roadmap.md"));

        Assert.True(article.Length >= 7000, $"Pose tutorial is too short: {article.Length} characters.");
        Assert.Contains("| 76 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("YoloVision Pose 教程", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);
        foreach (string marker in new[]
        {
            "## 当前实现范围",
            "YoloDetection.SourceIndex",
            "[1,N,K*stride]",
            "[1,K*stride,N]",
            "stride 为 2",
            "score 设为 `1.0`",
            "不推断骨架连接",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-pose",
            "--exportReport",
            "--output-role-map boxes:det,keypoints:pose-keypoints",
            "--pose-keypoint-count 17",
            "--keypoint-stride 3",
            "--aux-layout boxes-first",
            "--preflight --strict-preflight",
            "--preprocessed-output",
            "--output-json",
            "--visualization-svg",
            "Test-YoloVisionOutputReport.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YoloPoseDecoder.cs",
            "real-model-runtime",
            "package-consumer-runtime",
            "blocked-by-cuda-driver",
            "## 收尾清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("--exportProfile", article, StringComparison.Ordinal);
        Assert.DoesNotContain("已经实现人体骨架", article, StringComparison.Ordinal);
    }

    [Fact]
    public void ObbTutorialIsLongFormAndDocumentsEmbeddedAnglesAndRotatedNmsEvidence()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-obb-tutorial.md"));
        string roadmap = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "technical-article-roadmap.md"));

        Assert.True(article.Length >= 7000, $"OBB tutorial is too short: {article.Length} characters.");
        Assert.Contains("| 77 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("YoloVision OBB 教程", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);
        foreach (string marker in new[]
        {
            "## 当前实现范围",
            "probabilistic-IoU rotated NMS",
            "Fast-NMS",
            "YoloDetection.SourceIndex",
            "[1,C,N]",
            "[1,N,1]",
            "[1,1,N]",
            "--angle-degrees",
            "--angle-radians",
            "AngleRadians",
            "angleUnit=radian",
            "angleRange=owner-record-required",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-obb",
            "--exportReport",
            "--output-role-map boxes:det,angles:obb-angle",
            "--obb-angle-output angles",
            "--aux-layout boxes-first",
            "--aux-channel-start 19",
            "[1,20,21504]",
            "0.997781",
            "Mismatches=1",
            "--preflight --strict-preflight",
            "--output-json",
            "--visualization-svg",
            "Test-YoloVisionOutputReport.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YoloObbDecoder.cs",
            "real-model-runtime",
            "package-consumer-runtime",
            "blocked-by-cuda-driver",
            "## 收尾清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("--exportProfile", article, StringComparison.Ordinal);
        Assert.DoesNotContain("没有实现 rotated-IoU NMS", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("当前是 axis-aligned NMS", article, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ClassificationSemanticTutorialBindsCodeCommandsReportsAndProofBoundaries()
    {
        string articlePath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-classification-semantic-tutorial.md");
        string article = File.ReadAllText(articlePath);
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string roadmap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("yolovision-classification-semantic-tutorial.md", readme, StringComparison.Ordinal);
        Assert.Contains("| 78 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "## 先判断任务",
            "## 全链路",
            "E:\\TensorRtSharpAssets\\cases\\cls-sem",
            "## Classification 输出契约",
            "[C,1]",
            "classification-score-mode logits",
            "## Semantic 输出契约",
            "[1,H,W,C]",
            "class-major",
            "32 列、24 行",
            "TensorRtExec build-only",
            "--exportReport",
            "--classification-output output0",
            "--classification-score-mode probabilities",
            "--semantic-output semantic",
            "--output-json",
            "--visualization-svg",
            "yolovision-output-cls.example.json",
            "yolovision-output-sem.example.json",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "samples/assets/yolovision-article-case-pack.json",
            "Test-YoloVisionOutputReport.ps1",
            "Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YoloVision Passed=True",
            "以下材料不得替代真实模型证明",
            "package-consumer-runtime",
            "blocked-by-cuda-driver",
            "## 发布前检查清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("YoloSampleRunner.DecodeClassifications", article, StringComparison.Ordinal);
        Assert.Contains("YoloSampleRunner.DecodeSemanticMap", article, StringComparison.Ordinal);
        Assert.Contains("YoloVisionVisualizationWriter", article, StringComparison.Ordinal);
        Assert.DoesNotContain("--semantic-map-shape", article, StringComparison.Ordinal);
        Assert.DoesNotContain("--palette ", article, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloFamilyProfileGuideIsPublishableLongFormAndBindsEachTaskToEvidence()
    {
        string articlePath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolo-family-profile-and-postprocess-guide.md");
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        foreach (string marker in new[]
        {
            "## 全链路",
            "E:\\TensorRtSharpAssets\\yolo-cases",
            "## 六任务接入矩阵",
            "## 每个任务的命令骨架",
            "TensorRtExec build-only",
            "samples/YoloVision/yolovision-task-output-contract.json",
            "samples/assets/yolovision-assets.template.json",
            "--task cls",
            "--task seg",
            "--task obb",
            "--task pose",
            "--task sem",
            "Test-YoloVisionOutputReport.ps1",
            "YoloVision Passed=True",
            "sample-run evidence",
            "package-consumer-runtime",
            "blocked-by-cuda-driver",
            "## 发布前检查清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains("`" + task + "`", article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("以下材料不得替代真实模型证明", article, StringComparison.Ordinal);
        Assert.Contains("local feed", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
        Assert.Contains("direct `.nupkg`", article, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionModelMatrixDocumentsFamiliesTasksAssetsAndProofBoundary()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolo-vision-model-matrix.md");
        string article = File.ReadAllText(articlePath);
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("articles/zh-cn/yolo-vision-model-matrix.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolo-vision-model-matrix.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("YoloVision", readme, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", readme, StringComparison.Ordinal);

        string combined = article + Environment.NewLine + readme;
        foreach (string family in new[] { "YOLOv5", "YOLOv6", "YOLOv7", "YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLOv26" })
        {
            Assert.Contains(family, combined, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, combined, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string required in new[]
        {
            "模型来源",
            "ONNX 导出",
            "输入",
            "输出 tensor",
            "后处理",
            "not-proof",
            "planned",
            "package-consumer-runtime",
            "real-model-runtime",
            "YoloVision Passed=True",
            "SHA256",
            "许可证"
        })
        {
            Assert.Contains(required, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("samples/assets/yolovision-assets.template.json", article, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar.yolovision.template.json", article, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.yolovision.template.json", article, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionTaskOutputContractKeepsTasksDocsAndOwnerPacksAligned()
    {
        string contractPath = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "yolovision-task-output-contract.json");
        string matrixPath = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "yolo-model-matrix.json");
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolo-vision-model-matrix.md"));
        string ownerBackfillPack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.json"));
        string articleCasePack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-article-case-pack.json"));

        Assert.True(File.Exists(contractPath));
        Assert.Contains("yolovision-task-output-contract.json", readme, StringComparison.Ordinal);
        Assert.Contains("yolovision-task-output-contract.json", article, StringComparison.Ordinal);

        using JsonDocument contract = JsonDocument.Parse(File.ReadAllText(contractPath));
        using JsonDocument matrix = JsonDocument.Parse(File.ReadAllText(matrixPath));
        JsonElement root = contract.RootElement;

        Assert.Equal("yolovision-task-output-contract", root.GetProperty("contractId").GetString());
        Assert.Contains("not real-model-runtime proof", root.GetProperty("proofBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package-consumer-runtime proof", root.GetProperty("proofBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("TensorRtExec report", root.GetProperty("forbiddenSubstitutes").GetRawText(), StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", root.GetProperty("forbiddenSubstitutes").GetRawText(), StringComparison.Ordinal);

        string[] matrixTasks = matrix.RootElement.GetProperty("tasks").EnumerateArray()
            .Select(static item => item.GetString()!)
            .OrderBy(static item => item, StringComparer.Ordinal)
            .ToArray();
        string[] contractTasks = root.GetProperty("tasks").EnumerateArray()
            .Select(static item => item.GetProperty("task").GetString()!)
            .OrderBy(static item => item, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(matrixTasks, contractTasks);

        foreach (JsonElement task in root.GetProperty("tasks").EnumerateArray())
        {
            string taskName = task.GetProperty("task").GetString()!;
            Assert.Contains(taskName, readme, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(taskName, article, StringComparison.OrdinalIgnoreCase);
            Assert.True(task.GetProperty("primaryOutputRoles").GetArrayLength() >= 1, taskName);
            Assert.True(task.GetProperty("requiredMetadata").GetArrayLength() >= 4, taskName);
            Assert.True(task.GetProperty("managedSurface").GetArrayLength() >= 1, taskName);
            Assert.Contains("--minShapes", task.GetProperty("tensorRtExecProfileHint").GetString()!, StringComparison.Ordinal);
            Assert.Contains("not", task.GetProperty("evidenceBoundary").GetString()!, StringComparison.OrdinalIgnoreCase);
            Assert.False(task.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(task.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

            if (taskName is "det" or "seg" or "pose" or "obb" or "cls")
            {
                Assert.Contains("\"task\": \"" + taskName + "\"", ownerBackfillPack, StringComparison.Ordinal);
            }

            Assert.Contains("\"task\": \"" + taskName + "\"", articleCasePack, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void YoloV10EndToEndGuideMatchesManagedDecoderAndSourceTreeRuntimeProofBoundary()
    {
        string relativeArticlePath = "articles/zh-cn/yolovision-yolov10-end-to-end-output-guide.md";
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov10-end-to-end-output-guide.md");
        string article = File.ReadAllText(articlePath);
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string decoder = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "YoloDetectionDecoder.cs"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains(relativeArticlePath, docsIndex, StringComparison.Ordinal);
        Assert.Contains(relativeArticlePath, docsToc, StringComparison.Ordinal);
        Assert.Contains("yolovision-yolov10-end-to-end-output-guide.md", readme, StringComparison.Ordinal);
        Assert.Contains("DecodeEndToEnd", decoder, StringComparison.Ordinal);

        string combined = article + Environment.NewLine + readme;
        foreach (string marker in new[]
        {
            "https://github.com/THU-MIG/yolov10",
            "[1,N,6]",
            "x1,y1,x2,y2,score,classId",
            "--layout end2end",
            "TensorRtExec",
            "YoloVision Passed=True",
            "SHA256",
            "7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3",
            "21891d0dcfb322069f864b5395f1653182251c35e4d822d2cc2c02635a4d2000",
            "38aaddac4e6f22f6d230d6e36c9e787c4d407508d1cb5dea768d0967827a8c4f",
            "bb2c5958590c4ac074b969d90af87f3aedb38991054588434133a2f620ebc43e",
            "dog=0.91683036",
            "output0:[1,300,6]",
            "AGPL-3.0-only",
            "许可证",
            "no second NMS",
            "not package-consumer-runtime"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument matrix = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "YoloVision",
            "yolo-model-matrix.json")));
        JsonElement yoloV10 = matrix.RootElement.GetProperty("entries").EnumerateArray()
            .Single(static entry => entry.GetProperty("family").GetString() == "yolov10");
        Assert.Contains("source-tree real-model runtime", yoloV10.GetProperty("status").GetString(), StringComparison.Ordinal);
        Assert.Contains("never applies a second NMS", yoloV10.GetProperty("postprocessNotes").GetString(), StringComparison.Ordinal);
    }
}
