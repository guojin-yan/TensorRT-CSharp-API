using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleCampaignFourthBatchBodyTests
{
    private static readonly string[] ArticleFiles =
    {
        "yolovision-detection-yolov8n-download-export-run.md",
        "yolovision-segmentation-mask-postprocess-guide.md",
        "yolovision-pose-keypoint-output-guide.md",
        "yolovision-obb-angle-output-guide.md",
        "tensorrtexec-winforms-screenshot-walkthrough.md",
        "tensorrtexec-report-schema-guide.md",
        "runtime-package-windows-linux-install-faq.md",
        "plugin-registry-inventory-user-guide.md",
        "deferred-readonly-api-upgrade-playbook.md",
        "csharp-wrapper-lifetime-design.md",
        "release-evidence-non-substitute-guide.md",
        "project-roadmap-to-public-release.md",
    };

    [Fact]
    public void FourthCampaignArticleBodyBatchExistsAndKeepsProofBoundaries()
    {
        Assert.InRange(ArticleFiles.Length, 10, 15);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articleFile in ArticleFiles)
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string href = "articles/zh-cn/" + articleFile;
            string readmePath = "docs/articles/zh-cn/" + articleFile;

            Assert.True(File.Exists(articlePath), articlePath);

            string article = File.ReadAllText(articlePath);
            Assert.Contains("适用读者", article, StringComparison.Ordinal);
            Assert.Contains("解决问题", article, StringComparison.Ordinal);
            Assert.Contains("背景与场景", article, StringComparison.Ordinal);
            Assert.True(
                article.Contains("操作路径", StringComparison.Ordinal) ||
                article.Contains("实现路径", StringComparison.Ordinal),
                articleFile + " must contain 操作路径 or 实现路径.");
            Assert.Contains("代码与文件入口", article, StringComparison.Ordinal);
            Assert.Contains("图示建议", article, StringComparison.Ordinal);
            Assert.Contains("边界说明", article, StringComparison.Ordinal);
            Assert.Contains("下一步", article, StringComparison.Ordinal);
            Assert.Contains("proof", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("runtime proof", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("build-only", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("dry-run", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("template", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("local feed", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
            Assert.Contains("direct `.nupkg`", article, StringComparison.Ordinal);
            Assert.Contains("TensorRtExec report", article, StringComparison.Ordinal);
            Assert.Contains("YoloVision matrix", article, StringComparison.Ordinal);
            Assert.Contains("OnnxToEngine report", article, StringComparison.Ordinal);
            Assert.Contains("readonly diagnostics", article, StringComparison.Ordinal);
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPromoteRuntimeProof=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);

            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains(readmePath, readme, StringComparison.Ordinal);
            Assert.Contains(readmePath, zhReadme, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void YoloVisionYolov8nDetectionArticleCoversAssetWorkspacePreprocessOutputValidationAndCandidateFields()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-detection-yolov8n-download-export-run.md"));

        foreach (string marker in new[]
        {
            "可复用资产目录与完整运行产物",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\models",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\labels",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\images",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\tensors",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\engines",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\reports",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-det\\logs",
            "yolovision-yolov8-det-candidate.template.json",
            "model.sourceUrl",
            "model.downloadUrl",
            "model.licenseEvidence",
            "model.onnxExportCommand",
            "Get-FileHash -Algorithm SHA256",
            "--preprocess-only",
            "--tensor-layout NCHW",
            "--color-order RGB",
            "--resize letterbox",
            "--output-json",
            "--visualization-svg",
            "Test-YoloVisionOutputReport.ps1",
            "Test-YoloVisionRealAssetCandidate.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "proofChecklist.requiredEvidenceLines",
            "proofChecklist.requiredHashes",
            "stdoutSummary",
            "stderrSummary",
            "packageConsumerBoundary",
            "仍不是 package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void YoloVisionYolov8nSegmentationArticleCoversPrototypeWorkspaceRolesHashesAndValidation()
    {
        string content = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "yolovision-segmentation-mask-postprocess-guide.md"));

        foreach (string marker in new[]
        {
            "可复用资产目录与完整验证",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\models",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\labels",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\images",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\tensors",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\engines",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\reports",
            "E:\\TensorRtSharpAssets\\cases\\yolov8n-seg\\logs",
            "yolovision-yolov8-seg-candidate.template.json",
            "model.sourceUrl",
            "model.downloadUrl",
            "model.license",
            "labels.sha256",
            "input.imageSha256",
            "input.preprocessedTensorSha256",
            "outputMetadata.outputRoleMap",
            "outputMetadata.prototypeShape",
            "outputMetadata.maskCoefficientCount",
            "outputMetadata.maskResizePolicy",
            "Get-FileHash -Algorithm SHA256",
            "--preprocess-only",
            "--tensor-layout NCHW",
            "--color-order RGB",
            "--resize letterbox",
            "--output-role-map boxes:det,proto:mask-prototypes",
            "--mask-coefficient-count 32",
            "--output-json",
            "--visualization-svg",
            "detection output shape",
            "prototype shape",
            "maskThreshold",
            "letterboxScale",
            "maskPixelCount",
            "boxBeforeCrop",
            "boxAfterResize",
            "Test-YoloVisionOutputReport.ps1",
            "Test-YoloVisionRealAssetCandidate.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "YoloVision Passed=True",
            "仍不是 package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, content, StringComparison.Ordinal);
        }
    }
}
