using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredNextReadonlyCandidateListWorkQueueTests
{
    [Fact]
    public void DeferredNextReadonlyCandidateListAnchorsFutureWorkToBTreeQueueWithoutRepeatingCompletedBatches()
    {
        string articlePath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "deferred-next-readonly-candidate-list.md");

        string article = File.ReadAllText(articlePath);
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("articles/zh-cn/deferred-next-readonly-candidate-list.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/deferred-next-readonly-candidate-list.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("deferred-btier-implementation-work-package.json", article, StringComparison.Ordinal);
        Assert.Contains("DeferredBTierWorkItemProofClosureLedgerTests.cs", article, StringComparison.Ordinal);
        Assert.Contains("DeferredBTier41To45ProofClosureTests.cs", article, StringComparison.Ordinal);
        Assert.Contains("DeferredBTier46To50ProofClosureTests.cs", article, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "btier-001",
            "btier-012",
            "btier-013",
            "btier-024",
            "btier-025",
            "btier-040",
            "IDimensionExpr",
            "IAlgorithm",
            "IErrorRecorder",
            "IPluginCreatorV3One",
            "IStreamReader",
            "IStreamWriter"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        foreach (string forbiddenPromotion in new[]
        {
            "canPromoteReleaseProof=true",
            "canDeleteDeferredRecord=true",
            "public IntPtr",
            "public nint"
        })
        {
            Assert.DoesNotContain(forbiddenPromotion, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Contains("不能删除 deferred manifest 来制造完成度", article, StringComparison.Ordinal);

        foreach (string boundary in new[]
        {
            "runtime proof",
            "package-consumer",
            "readonly diagnostics",
            "build-only",
            "dry-run",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(boundary, article, StringComparison.OrdinalIgnoreCase);
        }
    }
}
