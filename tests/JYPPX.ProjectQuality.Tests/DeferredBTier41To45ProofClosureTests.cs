using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTier41To45ProofClosureTests
{
    [Fact]
    public void Btier41To45KeepSafeAlternativesDeferredHistoryAndPointerFreeWrappers()
    {
        string workPackage = ReadSource("artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        using JsonDocument document = JsonDocument.Parse(workPackage);
        Dictionary<string, JsonElement> workItems = document.RootElement.GetProperty("workItems")
            .EnumerateArray()
            .Where(static item => item.GetProperty("workItemId").GetString() is "btier-041" or "btier-042" or "btier-043" or "btier-044" or "btier-045")
            .ToDictionary(static item => item.GetProperty("workItemId").GetString()!, static item => item);

        Assert.Equal(5, workItems.Count);
        foreach (JsonElement item in workItems.Values)
        {
            Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
            Assert.Equal("phase-1-safe-alternative-proof", item.GetProperty("phase").GetString());
            Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
            Assert.NotEmpty(item.GetProperty("safeAlternativeManifestIds").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("deferredHistoryManifestIds").EnumerateArray());
        }

        string builderConfigApi = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs");
        string engineApi = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string contextApi = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs");
        string onnxParserApi = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs");
        string parserRefitterApi = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.cs");
        string parserRefitterSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs");
        string interopRuntimeControls =
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Builder", "NativeBridgeApi.BuilderConfigRuntimeControls.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Engine", "NativeBridgeApi.EngineRuntimeControls.cs");
        string interopDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Execution", "NativeBridgeApi.ExecutionContextDiagnostics.cs");
        string interopParserRefitter = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.ParserRefitterDiagnostics.cs");
        string managedSurface = string.Join('\n', builderConfigApi, engineApi, contextApi, onnxParserApi, parserRefitterApi, parserRefitterSnapshot);

        Assert.Contains("public TensorRtTilingOptimizationLevel GetTilingOptimizationLevel()", builderConfigApi);
        Assert.Contains("NativeBridgeApi.GetBuilderConfigTilingOptimizationLevel(Line, _handle)", builderConfigApi);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_tiling_optimization_level", interopRuntimeControls);

        Assert.Contains("public bool HasImplicitBatchDimensionCompatibility", engineApi);
        Assert.Contains("NativeBridgeApi.HasEngineImplicitBatchDimensionCompatibility(Line, _handle)", engineApi);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_cuda_engine_has_implicit_batch_dimension", interopRuntimeControls);

        Assert.Contains("public TensorRtProfilingVerbosity GetNvtxVerbosity()", contextApi);
        Assert.Contains("NativeBridgeApi.GetExecutionContextNvtxVerbosity(Line, _handle)", contextApi);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_nvtx_verbosity", interopDiagnostics);

        Assert.Contains("public TensorRtParserErrorInfo GetError(int index)", onnxParserApi);
        Assert.Contains("public IReadOnlyList<TensorRtOnnxParserDiagnostic> GetDiagnostics()", onnxParserApi);
        Assert.Contains("ReadOnnxParserErrorString(line, parser, index", ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.OnnxParserDiagnostics.cs"));

        Assert.Contains("public TensorRtParserErrorInfo GetError(int index)", parserRefitterApi);
        Assert.Contains("public TensorRtOnnxParserRefitterDiagnosticSnapshot GetDiagnosticSnapshot()", parserRefitterApi);
        Assert.Contains("Variable-length strings are copied through caller-owned buffers", parserRefitterApi);
        Assert.Contains("public TensorRtOnnxParserRefitterDiagnosticSummary ToSummary()", parserRefitterSnapshot);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_get_error", interopParserRefitter);

        Assert.DoesNotContain("public IntPtr", managedSurface);
        Assert.DoesNotContain("public nint", managedSurface);
    }

    [Fact]
    public void Btier41To45NativeAndDocsKeepProofBoundaryExplicit()
    {
        string trt10Header = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string trt10Api = string.Join(
            '\n',
            ReadSource("native", "src", "tensorrt", "v10", "api.cpp"),
            ReadSource("native", "src", "tensorrt", "v10", "modules", "builder", "builder_config.inc"));
        string trt10Deferred = string.Join(
            '\n',
            ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json"),
            ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-fourth-batch-onnx-parser-global-deferred.manifest.json"));
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string closureArticle = ReadSource("docs", "articles", "zh-cn", "deferred-btier-41-45-proof-closure.md");
        string nextList = ReadSource("docs", "articles", "zh-cn", "deferred-next-readonly-candidate-list.md");
        string docsToc = ReadSource("docs", "toc.yml");

        Assert.Contains("jyppx_trt10_builder_config_get_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)", trt10Header);
        Assert.Contains("jyppx_trt10_cuda_engine_has_implicit_batch_dimension(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_implicit_batch)", trt10Header);
        Assert.Contains("jyppx_trt10_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity)", trt10Header);
        Assert.Contains("jyppx_trt10_onnx_parser_get_error(JYPPX_TensorRtOnnxParser* parser, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)", trt10Header);
        Assert.Contains("jyppx_trt10_parser_refitter_get_error(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)", trt10Header);

        Assert.Contains("config_payload->getTilingOptimizationLevel()", trt10Api);
        Assert.Contains("engine_payload->hasImplicitBatchDimension()", trt10Api);
        Assert.Contains("context_payload->getNvtxVerbosity()", trt10Api);
        Assert.Contains("fill_parser_refitter_error(parser_error, index, out_error);", ReadSource("native", "src", "tensorrt", "common", "parser_refitter_diagnostics.inc"));

        Assert.Contains("trt10-builder-config-get-tiling-optimization-level-deferred", trt10Deferred);
        Assert.Contains("trt10-cuda-engine-has-implicit-batch-dimension-deferred", trt10Deferred);
        Assert.Contains("trt10-execution-context-get-nvtx-verbosity-deferred", trt10Deferred);
        Assert.Contains("trt10-parser-refitter-get-error-deferred", trt10Deferred);

        foreach (string id in new[] { "btier-041", "btier-042", "btier-043", "btier-044", "btier-045" })
        {
            Assert.Contains(id, manualGroups);
            Assert.Contains(id, closureArticle);
        }

        Assert.Contains("Deferred B-tier 41-45 proof closure", closureArticle);
        Assert.Contains("`btier-041` 到 `btier-045`", closureArticle);
        Assert.Contains("canDeleteDeferredRecord=false", closureArticle);
        Assert.Contains("canPromoteReleaseProof=false", closureArticle);
        Assert.Contains("runtime proof", closureArticle);
        Assert.Contains("第四批", nextList);
        Assert.Contains("Deferred BTier 41 45 Proof Closure", docsToc);
    }

    private static void AssertWorkItem(JsonElement item, string interfaceName, string safeAlternativeId, string deferredHistoryId)
    {
        Assert.Equal(interfaceName, item.GetProperty("interface").GetString());
        Assert.Equal("phase-2-wrapper-docs-quality-proof", item.GetProperty("phase").GetString());
        Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
        Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
        Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.Contains(safeAlternativeId, item.GetProperty("safeAlternativeManifestIds").EnumerateArray().Select(static value => value.GetString()!));
        Assert.Contains(deferredHistoryId, item.GetProperty("deferredHistoryManifestIds").EnumerateArray().Select(static value => value.GetString()!));
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
