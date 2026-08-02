using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionRealAssetCandidatePackTests
{
    [Fact]
    public void YoloVisionFamilyTaskRoadmapCoversBroadYoloSeriesWithoutPromotingProof()
    {
        string roadmapPath = Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-family-task-real-asset-roadmap.json");
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-family-task-real-asset-roadmap.md");
        string assetsReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "README.md"));
        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(roadmapPath));
        Assert.True(File.Exists(articlePath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(roadmapPath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-family-task-real-asset-roadmap", root.GetProperty("recordKind").GetString());
        Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("roadmapState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("not real-model-runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);

        JsonElement[] families = root.GetProperty("families").EnumerateArray().ToArray();
        Assert.Equal(9, families.Length);
        foreach (string family in new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom" })
        {
            JsonElement entry = Assert.Single(families, item => item.GetProperty("family").GetString() == family);
            Assert.Equal("owner-action-required", entry.GetProperty("runtimeProofState").GetString());
            Assert.True(entry.GetProperty("primaryTasks").GetArrayLength() >= 1);
            Assert.Contains("TensorRtExec", entry.GetProperty("tensorRtExecBuildCommandTemplate").GetString(), StringComparison.Ordinal);
            Assert.Contains("samples\\YoloVision", entry.GetProperty("yoloVisionRunCommandTemplate").GetString(), StringComparison.Ordinal);
            Assert.Contains("owner", entry.GetProperty("exportCommandTemplate").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.True(entry.GetProperty("requiredOutputMetadata").GetArrayLength() >= 1);
            Assert.True(entry.GetProperty("articleAngles").GetArrayLength() >= 1);
        }

        JsonElement yoloV8 = Assert.Single(families, item => item.GetProperty("family").GetString() == "yolov8");
        string[] yoloV8Tasks = yoloV8.GetProperty("primaryTasks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(new[] { "det", "cls", "seg", "obb", "pose", "sem" }, yoloV8Tasks);

        string[] requiredEvidence = root.GetProperty("requiredOwnerEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(requiredEvidence, static item => item.Contains("YoloVision run command", StringComparison.Ordinal));
        Assert.Contains(requiredEvidence, static item => item.Contains("YoloVision Passed=True", StringComparison.Ordinal));
        Assert.Contains(requiredEvidence, static item => item.Contains("owner review", StringComparison.OrdinalIgnoreCase));

        string[] forbiddenSubstitutes = root.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("TensorRtExec build-only report", forbiddenSubstitutes);
        Assert.Contains("YoloVision matrix", forbiddenSubstitutes);
        Assert.Contains("sidecar-only report", forbiddenSubstitutes);
        Assert.Contains("ProjectReference", forbiddenSubstitutes);

        string article = File.ReadAllText(articlePath);
        Assert.Contains("YOLOv5", article, StringComparison.Ordinal);
        Assert.Contains("YOLOv26", article, StringComparison.Ordinal);
        Assert.Contains("不是 `real-model-runtime` proof", article, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", article, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-family-task-real-asset-roadmap.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-family-task-real-asset-roadmap.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("yolovision-family-task-real-asset-roadmap.json", assetsReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-family-task-real-asset-roadmap.json", yoloReadme, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", article + assetsReadme + yoloReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionYoloV8CandidateTemplatesCaptureAssetsCommandsAndProofBoundaries()
    {
        string[] templateFiles =
        {
            "yolovision-yolov8-det-candidate.template.json",
            "yolovision-yolov8-seg-candidate.template.json",
            "yolovision-yolov8-pose-candidate.template.json",
            "yolovision-yolov8-obb-candidate.template.json",
            "yolovision-yolov8-cls-candidate.template.json",
            "yolovision-yolov8-sem-candidate.template.json"
        };

        foreach (string templateFile in templateFiles)
        {
            string path = Path.Combine(RepositoryPaths.Root, "samples", "assets", templateFile);
            Assert.True(File.Exists(path), templateFile);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            string task = root.GetProperty("task").GetString()!;

            Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
            Assert.Equal("YOLOv8", root.GetProperty("family").GetString());
            Assert.True(task is "det" or "seg" or "pose" or "obb" or "cls" or "sem");
            Assert.Equal("owner-action-required", root.GetProperty("runtimeProofState").GetString());
            Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());
            Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

            JsonElement model = root.GetProperty("model");
            Assert.Contains("YOLOv8", model.GetProperty("name").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("yolo export", model.GetProperty("onnxExportCommand").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Equal("owner-required", model.GetProperty("sha256").GetString());
            Assert.Equal("owner-required", model.GetProperty("license").GetString());

            JsonElement input = root.GetProperty("input");
            string expectedInputShape = task switch
            {
                "cls" => "1x3x224x224",
                "obb" => "1x3x1024x1024",
                "sem" => "1x3x512x512",
                _ => "1x3x640x640",
            };
            Assert.Equal(expectedInputShape, input.GetProperty("inputShape").GetString());
            Assert.Equal("owner-required", input.GetProperty("imageSha256").GetString());
            Assert.Equal("owner-required", input.GetProperty("preprocessedTensorSha256").GetString());

            JsonElement commands = root.GetProperty("commands");
            Assert.Contains("applications\\TensorRtExec", commands.GetProperty("tensorRtExecBuildCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("samples\\YoloVision", commands.GetProperty("yoloVisionRunCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("--buildOnly", commands.GetProperty("tensorRtExecBuildCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("--task " + task, commands.GetProperty("yoloVisionRunCommand").GetString(), StringComparison.Ordinal);
            Assert.Equal("samples/YoloVision/yolovision-task-output-contract.json", root.GetProperty("taskOutputContract").GetString());

            JsonElement outputMetadata = root.GetProperty("outputMetadata");
            if (task == "pose")
            {
                Assert.Equal(17, outputMetadata.GetProperty("keypointCount").GetInt32());
                Assert.Equal("x,y,confidence", outputMetadata.GetProperty("keypointLayout").GetString());
                Assert.Equal("confidence", outputMetadata.GetProperty("keypointScoreField").GetString());
            }
            else if (task == "obb")
            {
                Assert.Contains("angle", outputMetadata.GetProperty("rotatedBoxLayout").GetString(), StringComparison.OrdinalIgnoreCase);
                Assert.True(outputMetadata.TryGetProperty("angleUnit", out _));
                Assert.True(outputMetadata.TryGetProperty("coordinateSpace", out _));
            }
            else if (task == "cls")
            {
                Assert.Equal(5, outputMetadata.GetProperty("topK").GetInt32());
                Assert.True(outputMetadata.GetProperty("labelsRequired").GetBoolean());
                Assert.True(outputMetadata.TryGetProperty("classScoreField", out _));
            }
            else if (task == "sem")
            {
                Assert.True(outputMetadata.TryGetProperty("semanticMapShape", out _));
                Assert.True(outputMetadata.TryGetProperty("classMapLayout", out _));
                Assert.True(outputMetadata.GetProperty("paletteRequired").GetBoolean());
            }

            JsonElement checklist = root.GetProperty("proofChecklist");
            string[] evidenceLines = checklist.GetProperty("requiredEvidenceLines").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains(evidenceLines, line => line.Contains("YoloVision Passed=True", StringComparison.Ordinal));
            Assert.Contains("package-consumer-runtime", checklist.GetProperty("packageConsumerBoundary").GetString(), StringComparison.Ordinal);
            Assert.Contains("owner backfills real assets", checklist.GetProperty("notProofUntil").GetString(), StringComparison.Ordinal);

            string[] requiredHashes = checklist.GetProperty("requiredHashes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("modelSha256", requiredHashes);
            Assert.Contains("labelsSha256", requiredHashes);
            Assert.Contains("imageSha256", requiredHashes);
            Assert.Contains("preprocessedTensorSha256", requiredHashes);
            Assert.Contains("runLogSha256", requiredHashes);
        }
    }

    [Fact]
    public void YoloVisionYoloV8TutorialsAreLinkedAndKeepTheirCurrentEvidenceBoundaries()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        foreach (string articleFile in new[]
        {
            "yolovision-yolov8-det-real-asset-tutorial.md",
            "yolovision-yolov8-seg-real-asset-tutorial.md",
            "yolovision-yolov8-seg-local-package-consumer-tutorial.md"
        })
        {
            string href = "articles/zh-cn/" + articleFile;
            string path = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(path);

            Assert.True(File.Exists(path));
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", article, StringComparison.Ordinal);
            Assert.Contains("SHA256", article, StringComparison.Ordinal);
        }

        string detection = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8-det-real-asset-tutorial.md"));
        Assert.Contains("real-model-runtime", detection, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", detection, StringComparison.Ordinal);
        Assert.Contains("705,600", detection, StringComparison.Ordinal);
        Assert.Contains("package-consumer", detection, StringComparison.Ordinal);
        Assert.Contains("公开再分发", detection, StringComparison.Ordinal);

        string segmentation = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8-seg-real-asset-tutorial.md"));
        Assert.Contains("real-model-runtime", segmentation, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", segmentation, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec", segmentation, StringComparison.Ordinal);
        Assert.Contains("仍不是公开 feed 或 post-publish 证明", segmentation, StringComparison.Ordinal);

        string packageConsumer = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8-seg-local-package-consumer-tutorial.md"));
        Assert.Contains("local-package-consumer-runtime", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("1,793,600", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("publicPackageProof", packageConsumer, StringComparison.Ordinal);
        Assert.Contains("ownerReleaseAcceptance", packageConsumer, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionArticleCasePackCoversSixPublishableYoloV8nCasesWithoutPromotingProof()
    {
        string packPath = Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-article-case-pack.json");
        string assetsReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "README.md"));
        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.True(File.Exists(packPath));
        Assert.Contains("yolovision-article-case-pack.json", assetsReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-article-case-pack.json", yoloReadme, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(packPath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-article-case-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("packState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("not real-model-runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);
        Assert.Equal("samples/YoloVision/yolovision-preflight.schema.json", root.GetProperty("preflightSchema").GetString());
        Assert.Equal("yolovision-preflight.v1", root.GetProperty("preflightSchemaVersion").GetString());
        Assert.Equal("precheck", root.GetProperty("preflightProofClassification").GetString());

        string[] requiredOwnerEvidence = root.GetProperty("requiredOwnerEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("modelSha256", requiredOwnerEvidence);
        Assert.Contains("labelsSha256", requiredOwnerEvidence);
        Assert.Contains("imageSha256", requiredOwnerEvidence);
        Assert.Contains("preprocessedTensorSha256", requiredOwnerEvidence);
        Assert.Contains(requiredOwnerEvidence, static item => item.Contains("offline preflight", StringComparison.OrdinalIgnoreCase));
        Assert.Contains("runLogSha256", requiredOwnerEvidence);
        Assert.Contains(requiredOwnerEvidence, static item => item.Contains("YoloVision Passed=True", StringComparison.Ordinal));

        string[] forbiddenSubstitutes = root.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("TensorRtExec report", forbiddenSubstitutes);
        Assert.Contains("YoloVision matrix", forbiddenSubstitutes);
        Assert.Contains("sidecar-only report", forbiddenSubstitutes);
        Assert.Contains("ProjectReference", forbiddenSubstitutes);

        JsonElement[] cases = root.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(6, cases.Length);
        Assert.Equal(new[] { "det", "seg", "pose", "obb", "cls", "sem" }, cases.Select(static item => item.GetProperty("task").GetString()).ToArray());

        foreach (JsonElement caseEntry in cases)
        {
            string task = caseEntry.GetProperty("task").GetString()!;
            string article = caseEntry.GetProperty("article").GetString()!;
            string href = article.StartsWith("docs/", StringComparison.Ordinal)
                ? article["docs/".Length..]
                : article;
            string articlePath = Path.Combine(RepositoryPaths.Root, article.Replace('/', Path.DirectorySeparatorChar));
            string articleText = File.ReadAllText(articlePath);

            Assert.True(File.Exists(articlePath), article);
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains("TensorRtExec", caseEntry.GetProperty("tensorRtExecBuildCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("--buildOnly", caseEntry.GetProperty("tensorRtExecBuildCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("samples\\YoloVision", caseEntry.GetProperty("yoloVisionRunCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("--task " + task, caseEntry.GetProperty("yoloVisionRunCommand").GetString(), StringComparison.Ordinal);
            Assert.Equal(task == "sem" ? "custom" : "v8", caseEntry.GetProperty("family").GetString());
            Assert.Contains("--preflight", caseEntry.GetProperty("yoloVisionPreflightCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("--preflight-report", caseEntry.GetProperty("yoloVisionPreflightCommand").GetString(), StringComparison.Ordinal);
            Assert.Equal("yolovision-preflight.v1", root.GetProperty("preflightSchemaVersion").GetString());
            Assert.Equal("models/yolov8n-" + task + "-preflight.json", caseEntry.GetProperty("yoloVisionPreflightReportPath").GetString());
            Assert.False(caseEntry.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.True(caseEntry.GetProperty("requiredOutputMetadata").GetArrayLength() >= 5);

            string[] evidenceLines = caseEntry.GetProperty("expectedEvidenceLines").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains(evidenceLines, static line => line.Contains("YoloVision Passed=True", StringComparison.Ordinal));

            Assert.Contains("YoloVision Passed=True", articleText, StringComparison.Ordinal);
            Assert.Contains("SHA256", articleText, StringComparison.Ordinal);
            Assert.Contains("TensorRtExec", articleText, StringComparison.Ordinal);
            Assert.Contains("Proof Boundary", articleText, StringComparison.Ordinal);
            Assert.Contains("Owner Backfill Checklist", articleText, StringComparison.Ordinal);
            Assert.Contains("不是 runtime proof", articleText, StringComparison.Ordinal);
            Assert.DoesNotContain("YoloDet", articleText, StringComparison.Ordinal);
        }

        JsonElement semantic = Assert.Single(cases, static item => item.GetProperty("task").GetString() == "sem");
        Assert.Contains("--semantic-output", semantic.GetProperty("yoloVisionRunCommand").GetString(), StringComparison.Ordinal);
        string[] semanticMetadata = semantic.GetProperty("requiredOutputMetadata").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("semanticMapShape", semanticMetadata);
        Assert.Contains("classMapLayout", semanticMetadata);
        Assert.Contains("paletteSha256", semanticMetadata);
    }
}
