using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredCompatibilityDiagnosticsProofTests
{
    [Fact]
    public void CompatibilityDiagnosticsBatchHasExistingPointerFreeRoutesAndSmokeMarkers()
    {
        string builderConfig = ReadSource("src", "JYPPX.TensorRtSharp", "Builder", "TensorRtBuilderConfig.Trt11RuntimeControls.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.cs");
        string legacyInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.LegacyParserDiagnostics.cs");
        string legacyWrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyParserDiagnostics.cs");
        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.Diagnostics.cs");
        string refitter = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitter.Diagnostics.cs");
        string parserInterop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.ParserRefitterDiagnostics.cs");
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
