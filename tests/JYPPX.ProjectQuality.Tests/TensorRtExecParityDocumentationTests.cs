using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecParityDocumentationTests
{
    [Fact]
    public void TensorRtExecTrtexecParityMatrixDocumentsRequiredCapabilitiesAndProofBoundary()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrt-exec-trtexec-parity-matrix.md");
        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string solution = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "TensorRtSharp.sln"));
        string appReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string commandSource = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs"));

        Assert.True(File.Exists(articlePath));
        Assert.Contains("articles/zh-cn/tensorrt-exec-trtexec-parity-matrix.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrt-exec-trtexec-parity-matrix.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("applications\\TensorRtExec\\TensorRtExec.csproj", solution, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "ONNX 模型输入",
            "engine 保存",
            "engine 加载",
            "dynamic shape",
            "min/opt/max shape profile",
            "timing iterations",
            "FP16",
            "INT8",
            "workspace / memory pool",
            "timing cache",
            "plugin library 参数边界",
            "profiling",
            "bounded benchmark scheduler",
            "layer dump",
            "report export alias",
            "verbose logging",
            "input/output binding metadata",
            "package consumer / runtime package key proof 边界"
        })
        {
            Assert.Contains(required, article, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string proofBoundary in new[]
        {
            "build-only",
            "parse-only",
            "dry-run",
            "dependency-probe-only",
            "sidecar-only",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "package-consumer-runtime"
        })
        {
            Assert.Contains(proofBoundary, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("--onnx", commandSource, StringComparison.Ordinal);
        Assert.Contains("--saveEngine", commandSource, StringComparison.Ordinal);
        Assert.Contains("--loadEngine", commandSource, StringComparison.Ordinal);
        Assert.Contains("--minShapes/--optShapes/--maxShapes", commandSource, StringComparison.Ordinal);
        Assert.Contains("--fp16 --int8", commandSource, StringComparison.Ordinal);
        Assert.Contains("--memPoolSize", commandSource, StringComparison.Ordinal);
        Assert.Contains("--timingCacheFile", commandSource, StringComparison.Ordinal);
        Assert.Contains("--plugins", commandSource, StringComparison.Ordinal);
        Assert.Contains("--exportProfile", commandSource, StringComparison.Ordinal);
        Assert.Contains("--exportLayerInfo", commandSource, StringComparison.Ordinal);
        Assert.Contains("--exportReport|--report", commandSource, StringComparison.Ordinal);
        Assert.Contains("--verbose", appReadme, StringComparison.Ordinal);
        Assert.Contains("BindingMetadata", appReadme, StringComparison.Ordinal);
        Assert.Contains("TensorRtExecReportFormatter", appReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", appReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecTimingCacheAndInt8OwnerFieldGuidesDocumentNonProofBoundaries()
    {
        string timingCachePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-timing-cache-owner-field-guide.md");
        string int8Path = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-int8-calibration-owner-field-guide.md");
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string timingCacheArticle = File.ReadAllText(timingCachePath);
        string int8Article = File.ReadAllText(int8Path);

        Assert.True(File.Exists(timingCachePath));
        Assert.True(File.Exists(int8Path));
        Assert.Contains("articles/zh-cn/tensorrtexec-timing-cache-owner-field-guide.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrtexec-timing-cache-owner-field-guide.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrtexec-int8-calibration-owner-field-guide.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/tensorrtexec-int8-calibration-owner-field-guide.md", docsToc, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "--timingCacheFile",
            "--exportTimingCache",
            "parse/report-only",
            "cache content hash",
            "native import/export smoke",
            "not runtime proof",
            "package-consumer-runtime"
        })
        {
            Assert.Contains(marker, timingCacheArticle, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "--int8",
            "--calib",
            "calibrator ownership",
            "calibration dataset provenance",
            "model-specific INT8 accuracy evidence",
            "callback ownership",
            "parse/report-only",
            "not runtime proof",
            "package-consumer-runtime"
        })
        {
            Assert.Contains(marker, int8Article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("canPromotePackageConsumerRuntime=true", timingCacheArticle + int8Article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("proofClassification=package-consumer-runtime", timingCacheArticle + int8Article, StringComparison.OrdinalIgnoreCase);
    }
}
