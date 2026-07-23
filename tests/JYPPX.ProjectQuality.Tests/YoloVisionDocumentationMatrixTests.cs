using Xunit;
using System.Text.Json;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionDocumentationMatrixTests
{
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
