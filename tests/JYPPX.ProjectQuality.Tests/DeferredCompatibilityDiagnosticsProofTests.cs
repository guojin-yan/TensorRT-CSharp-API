using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredCompatibilityDiagnosticsProofTests
{
    [Fact]
    public void CompatibilityDiagnosticsBatchKeepsExplicitAliasesAndVersionBoundaries()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string coverage = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-coverage.csv");
        string auditMarkdown = ReadSource("artifacts", "interface-coverage", "trt-compatibility-diagnostics-candidate-audit.md");
        string article = ReadSource("docs", "articles", "zh-cn", "deferred-compatibility-diagnostics-proof.md");
        string toc = ReadSource("docs", "toc.yml");
        using JsonDocument audit = JsonDocument.Parse(ReadSource("artifacts", "interface-coverage", "trt-compatibility-diagnostics-candidate-audit.json"));

        Assert.Contains("\"IBuilderConfig::getTilingOptimizationLevel\" = @(\"id:*builder-config-get-tiling-optimization-level-deferred\")", script);
        Assert.Contains("\"IBuilderConfig::setTilingOptimizationLevel\" = @(\"id:*builder-config-set-tiling-optimization-level-deferred\")", script);
        Assert.Contains("\"ICudaEngine::hasImplicitBatchDimension\" = @(\"id:*cuda-engine-has-implicit-batch-dimension-deferred\")", script);
        Assert.Contains("\"IParser::getError\" = @(\"id:*parser-refitter-get-error-deferred\")", script);
        Assert.Contains("\"IParserRefitter::getError\" = @(\"id:*parser-refitter-get-error-deferred\")", script);

        foreach (string marker in new[]
        {
            "IBuilderConfig::getTilingOptimizationLevel",
            "ICudaEngine::hasImplicitBatchDimension",
            "IUffParser::getUffRequiredVersionPatch",
            "IParser::getError",
            "IParserRefitter::getError",
            "TRT8",
            "TRT10",
            "TRT11",
            "vendor declarations",
            "import library/DLL",
            "deferred history remains",
        })
        {
            Assert.Contains(marker, auditMarkdown, StringComparison.Ordinal);
        }
        Assert.Contains("deferred-compatibility-diagnostics-proof.md", toc, StringComparison.Ordinal);
        Assert.Contains("canDeleteDeferredRecords=false", article, StringComparison.Ordinal);

        JsonElement root = audit.RootElement;
        Assert.Equal("trt-compatibility-diagnostics-candidate-audit", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.False(root.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.Equal(5, root.GetProperty("candidates").GetArrayLength());
        Assert.All(root.GetProperty("candidates").EnumerateArray(), static candidate =>
        {
            Assert.True(candidate.GetProperty("publicApiPointerFree").GetBoolean());
            Assert.False(candidate.GetProperty("canDeleteDeferredRecords").GetBoolean());
            Assert.NotEmpty(candidate.GetProperty("realManifestIds").EnumerateArray());
            Assert.NotEmpty(candidate.GetProperty("deferredHistoryManifestIds").EnumerateArray());
            Assert.NotEmpty(candidate.GetProperty("versionGuards").EnumerateArray());
        });

        Assert.Contains("\"IBuilderConfig\",\"getTilingOptimizationLevel\",\"IBuilderConfig::getTilingOptimizationLevel\",\"builder\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"ICudaEngine\",\"hasImplicitBatchDimension\",\"ICudaEngine::hasImplicitBatchDimension\",\"engine-context\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IUffParser\",\"getUffRequiredVersionPatch\",\"IUffParser::getUffRequiredVersionPatch\",\"legacy-parser\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IParser\",\"getError\",\"IParser::getError\",\"onnx-parser\",\"implemented-with-deferred-history\"", coverage);
        Assert.Contains("\"IParserRefitter\",\"getError\",\"IParserRefitter::getError\",\"onnx-parser\",\"implemented-with-deferred-history\"", coverage);
    }

    [Fact]
    public void CompatibilityDiagnosticsBatchHasExistingPointerFreeRoutesAndSmokeMarkers()
    {
        string builderConfig = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string legacyInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.LegacyParserDiagnostics.cs");
        string legacyWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyParserDiagnostics.cs");
        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.cs");
        string refitter = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.cs");
        string parserInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.ParserRefitterDiagnostics.cs");
        string networkSmoke = ReadSource("smoke", "NetworkBuilderSmokeRunner", "Program.cs");
        string legacySmoke = ReadSource("smoke", "LegacyParserDiagnosticsSmokeRunner", "Program.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("public TensorRtTilingOptimizationLevel GetTilingOptimizationLevel()", builderConfig);
        Assert.Contains("public bool HasImplicitBatchDimensionCompatibility", engine);
        Assert.Contains("public static TensorRtLegacyUffRequiredVersionSnapshot GetUffRequiredVersion", legacyWrapper);
        Assert.Contains("public TensorRtParserErrorInfo GetError(int index)", parser);
        Assert.Contains("public TensorRtOnnxParserRefitterDiagnosticSnapshot GetDiagnosticSnapshot()", refitter);
        Assert.Contains("parser_refitter_get_error", parserInterop);
        Assert.Contains("out int patch", legacyInterop);
        Assert.Contains("ProbeTilingControls(config)", networkSmoke);
        Assert.Contains("engine.HasImplicitBatchDimensionCompatibility", networkSmoke);
        Assert.Contains("UffRequiredVersion=", legacySmoke);
        Assert.Contains("ParserErrorSummary=", onnxSmoke);
        Assert.Contains("ParserRefitterDiagnosticSnapshot=", onnxSmoke);

        string managed = string.Join('\n', builderConfig, engine, legacyWrapper, parser, refitter, parserInterop, legacyInterop);
        Assert.DoesNotContain("public IntPtr", managed);
        Assert.DoesNotContain("public nint", managed);
        Assert.DoesNotContain("public UIntPtr", managed);
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
