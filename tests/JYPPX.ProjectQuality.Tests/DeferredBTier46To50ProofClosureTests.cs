using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTier46To50ProofClosureTests
{
    [Fact]
    public void WorkPackageItemsKeepSafeAlternativesAndDeferredHistory()
    {
        string workPackage = ReadSource("artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        using JsonDocument document = JsonDocument.Parse(workPackage);
        Dictionary<string, JsonElement> workItems = document.RootElement.GetProperty("workItems")
            .EnumerateArray()
            .Where(static item => item.GetProperty("workItemId").GetString() is "btier-046" or "btier-047" or "btier-048" or "btier-049" or "btier-050" or "btier-051")
            .ToDictionary(static item => item.GetProperty("workItemId").GetString()!, static item => item);

        Assert.Equal(6, workItems.Count);
        AssertWorkItem(workItems["btier-046"], "IUffParser::getUffRequiredVersionPatch", "trt8-legacy-uff-get-required-version", "trt8-uff-parser-get-uff-required-version-patch-deferred");
        AssertWorkItem(workItems["btier-047"], "IBuilderConfig::getTilingOptimizationLevel", "jyppx-trt10-builder-config-get-tiling-optimization-level", "trt10-builder-config-get-tiling-optimization-level-deferred");
        AssertWorkItem(workItems["btier-048"], "ICudaEngine::hasImplicitBatchDimension", "jyppx-trt10-cuda-engine-has-implicit-batch-dimension", "trt10-cuda-engine-has-implicit-batch-dimension-deferred");
        AssertWorkItem(workItems["btier-049"], "IExecutionContext::getNvtxVerbosity", "jyppx-trt10-execution-context-get-nvtx-verbosity", "trt10-execution-context-get-nvtx-verbosity-deferred");
        AssertWorkItem(workItems["btier-050"], "IParser::getError", "trt10-onnx-parser-get-error", "trt10-parser-refitter-get-error-deferred");
        AssertWorkItem(workItems["btier-051"], "IParserRefitter::getError", "trt10-parser-refitter-get-error", "trt10-parser-refitter-get-error-deferred");
    }

    [Fact]
    public void CoverageAliasesPromoteSafeRoutesAndRetainHistory()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("\"IBuilderConfig::getTilingOptimizationLevel\" = @(\"id:*builder-config-get-tiling-optimization-level\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IExecutionContext::getNvtxVerbosity\" = @(\"id:*execution-context-get-nvtx-verbosity\")", script, StringComparison.Ordinal);
        Assert.Contains("\"ICudaEngine::hasImplicitBatchDimension\" = @(\"id:*cuda-engine-has-implicit-batch-dimension\", \"id:*cuda-engine-has-implicit-batch-dimension-legacy-alias\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IBuilderConfig\",\"getTilingOptimizationLevel\",\"IBuilderConfig::getTilingOptimizationLevel\",\"builder\",\"implemented-with-deferred-history\"", coverage, StringComparison.Ordinal);
        Assert.Contains("jyppx-trt10-builder-config-get-tiling-optimization-level;trt10-builder-config-get-tiling-optimization-level-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("trt8-legacy-uff-get-required-version;trt8-uff-parser-get-uff-required-version-patch-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("trt10-parser-refitter-get-error;trt10-parser-refitter-get-error-count;trt10-parser-refitter-get-error-deferred", coverage, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicWrappersRemainPointerFreeAndDocsStateProofBoundary()
    {
        string wrappers = string.Join(
            '\n',
            ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.Diagnostics.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.Diagnostics.cs"));
        string article = ReadSource("docs", "articles", "zh-cn", "deferred-btier-46-50-proof-closure.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("GetTilingOptimizationLevel", wrappers, StringComparison.Ordinal);
        Assert.Contains("HasImplicitBatchDimensionCompatibility", wrappers, StringComparison.Ordinal);
        Assert.Contains("GetNvtxVerbosity", wrappers, StringComparison.Ordinal);
        Assert.Contains("GetError(int index)", wrappers, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", wrappers, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", wrappers, StringComparison.Ordinal);
        Assert.Contains("canDeleteDeferredRecord=false", article, StringComparison.Ordinal);
        Assert.Contains("canPromoteReleaseProof=false", article, StringComparison.Ordinal);
        Assert.Contains("deferred-btier-46-50-proof-closure.md", toc, StringComparison.Ordinal);
    }

    private static void AssertWorkItem(JsonElement item, string interfaceName, string safeAlternativeId, string deferredHistoryId)
    {
        Assert.Equal(interfaceName, item.GetProperty("interface").GetString());
        Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
        Assert.Contains(safeAlternativeId, item.GetProperty("safeAlternativeManifestIds").EnumerateArray().Select(static value => value.GetString()!));
        Assert.Contains(deferredHistoryId, item.GetProperty("deferredHistoryManifestIds").EnumerateArray().Select(static value => value.GetString()!));
        Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
        Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
