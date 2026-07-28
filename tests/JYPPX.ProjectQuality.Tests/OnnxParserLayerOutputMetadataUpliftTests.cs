using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxParserLayerOutputMetadataUpliftTests
{
    [Fact]
    public void ManifestsUseVersionedCallerOwnedCopiedMetadataContracts()
    {
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-onnx-parser-layer-output-metadata.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-onnx-parser-layer-output-metadata.manifest.json");

        AssertManifest(manifest10, "10");
        AssertManifest(manifest11, "11");
        Assert.Contains("trt11-parser-get-layer-output-tensor-deferred", ReadSource(
            "native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json"));
    }

    [Fact]
    public void TensorRt8OmitsTheUnavailableEntrypointWhileTenAndElevenDeclareIt()
    {
        string header8 = ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");

        Assert.DoesNotContain("onnx_parser_get_layer_output_tensor_metadata", header8, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt10_onnx_parser_get_layer_output_tensor_metadata", header10, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_onnx_parser_get_layer_output_tensor_metadata", header11, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeBoundaryCopiesEveryFieldAndResetsOutputsAcrossFailures()
    {
        string source = ReadSource("native", "src", "tensorrt", "common", "onnx_parser_support.inc");

        Assert.Contains("#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10", source, StringComparison.Ordinal);
        Assert.Contains("reset_onnx_parser_layer_output_tensor_metadata", source, StringComparison.Ordinal);
        Assert.Contains("parser->getLayerOutputTensor(layer_name, output_index)", source, StringComparison.Ordinal);
        Assert.Contains("tensor->getDimensions()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->getName()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->getType()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->getLocation()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->getAllowedFormats()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->isShapeTensor()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->isExecutionTensor()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->isNetworkInput()", source, StringComparison.Ordinal);
        Assert.Contains("tensor->isNetworkOutput()", source, StringComparison.Ordinal);
        Assert.Contains("dimensions.nbDims > 8", source, StringComparison.Ordinal);
        Assert.Contains("__try", source, StringComparison.Ordinal);
        Assert.Contains("report_vendor_seh_exception", source, StringComparison.Ordinal);
        Assert.Contains("catch (const std::exception& exception)", source, StringComparison.Ordinal);
        Assert.Contains("catch (...)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX_TensorRtTensor**", source, StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratedInteropAndManagedSurfaceRemainPointerFree()
    {
        string generated = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "GeneratedTensorRtManifestNativeMethods.g.cs");
        string interop = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxParserLayerOutputMetadata.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelSupport.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxLayerOutputTensorMetadata.cs");

        Assert.Contains("jyppx_trt10_onnx_parser_get_layer_output_tensor_metadata", generated, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_onnx_parser_get_layer_output_tensor_metadata", generated, StringComparison.Ordinal);
        Assert.Contains("byte[] output_buffer", generated, StringComparison.Ordinal);
        Assert.Contains("out NativeTensorRtDims64 out_shape", generated, StringComparison.Ordinal);
        Assert.Contains("ReadUtf8Buffer", interop, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt8", interop, StringComparison.Ordinal);
        Assert.Contains("BridgeStatusCode.NotSupported", interop, StringComparison.Ordinal);
        Assert.Contains("public bool TryGetLayerOutputTensorMetadata", wrapper, StringComparison.Ordinal);
        Assert.Contains("public TensorRtOnnxLayerOutputTensorMetadata GetLayerOutputTensorMetadata", wrapper, StringComparison.Ordinal);
        Assert.Contains("public bool PointerFreeCopiedMetadata => true", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool RetainsNativeTensor => false", snapshot, StringComparison.Ordinal);
        Assert.Contains("public bool CanDeleteDeferredRecord => false", snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", wrapper + snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", wrapper + snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", wrapper + snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", wrapper + snapshot, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeConsumerCoverageAndAuditKeepTheEvidenceBoundaryExplicit()
    {
        string smoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string auditText = ReadSource("artifacts", "interface-coverage", "trt-deferred-safe-uplift-candidate-audit.json");
        using JsonDocument audit = JsonDocument.Parse(auditText);

        JsonElement inventory = audit.RootElement.GetProperty("inventory");
        Assert.Equal(598, inventory.GetProperty("tensorRtDeferredRows").GetInt32());
        Assert.Equal(0, inventory.GetProperty("lowRiskRows").GetInt32());
        Assert.Equal(
            "owner-scoped-copied-metadata-safe-alternative",
            audit.RootElement.GetProperty("decision").GetString());
        Assert.Contains("TryGetLayerOutputTensorMetadata", smoke, StringComparison.Ordinal);
        Assert.Contains("GetLayerOutputTensorMetadata", smoke, StringComparison.Ordinal);
        Assert.Contains("jyppx_missing_layer", smoke, StringComparison.Ordinal);
        Assert.Contains("catch (Exception exception) when (IsSkippableEnvironmentException(exception))", smoke, StringComparison.Ordinal);
        Assert.Contains("Skipped=True Reason=ParserConstruction:", smoke, StringComparison.Ordinal);
        Assert.Contains("TensorRtOnnxLayerOutputTensorMetadata", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtOnnxParser.TryGetLayerOutputTensorMetadata)", consumer, StringComparison.Ordinal);
        Assert.Contains("onnx-parser-layer-output-copied-metadata", consumer, StringComparison.Ordinal);
        Assert.Contains(".IndexOf(\"/JYPPX.\", [System.StringComparison]::OrdinalIgnoreCase) -ge 0", consumer, StringComparison.Ordinal);
        Assert.Contains("id:*onnx-parser-get-layer-output-tensor-metadata", coverage, StringComparison.Ordinal);
        Assert.Contains("id:*onnx-parser-layer-output-tensor-exists", coverage, StringComparison.Ordinal);
        Assert.Contains("id:*parser-get-layer-output-tensor-deferred", coverage, StringComparison.Ordinal);

        string article = ReadSource("docs", "articles", "zh-cn", "onnx-parser-layer-output-copied-metadata.md");
        string toc = ReadSource("docs", "toc.yml");
        Assert.Contains("TryGetLayerOutputTensorMetadata", article, StringComparison.Ordinal);
        Assert.Contains("TRT11.0/CUDA12.9", article, StringComparison.Ordinal);
        Assert.Contains("onnx-parser-layer-output-copied-metadata.md", toc, StringComparison.Ordinal);
    }

    [Fact]
    public void CompactRuntimeEvidenceSeparatesPassedMetadataFromDependencyOnlyAndCompileProof()
    {
        string evidenceText = ReadSource(
            "artifacts", "interface-coverage", "onnx-parser-layer-output-metadata-runtime-evidence.json");
        using JsonDocument evidence = JsonDocument.Parse(evidenceText);
        JsonElement root = evidence.RootElement;

        Assert.Equal("copied-metadata-runtime-passed", root.GetProperty("tensorRt10").GetProperty("state").GetString());
        Assert.Equal("output", root.GetProperty("tensorRt10").GetProperty("tensorName").GetString());
        Assert.True(root.GetProperty("tensorRt10").GetProperty("pointerFreeCopiedMetadata").GetBoolean());
        Assert.False(root.GetProperty("tensorRt10").GetProperty("retainsNativeTensor").GetBoolean());
        Assert.Equal("parser-dependency-controlled-skip", root.GetProperty("tensorRt8").GetProperty("state").GetString());
        Assert.True(root.GetProperty("tensorRt8").GetProperty("controlledSkip").GetBoolean());
        Assert.False(root.GetProperty("tensorRt8").GetProperty("metadataQueryAttempted").GetBoolean());
        Assert.Equal("dependency-runtime-probe-only", root.GetProperty("tensorRt11").GetProperty("state").GetString());
        Assert.False(root.GetProperty("tensorRt11").GetProperty("metadataQueryAttempted").GetBoolean());
        Assert.Equal(3, root.GetProperty("packageConsumer").GetProperty("lines").GetArrayLength());
        Assert.Equal(
            "compile-surface-proof",
            root.GetProperty("packageConsumer").GetProperty("wrapperSurfaceEvidenceKind").GetString());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecord").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());

        string validator = ReadSource("eng", "Test-OnnxParserLayerOutputMetadataEvidence.ps1");
        Assert.Contains("OnnxParserLayerOutputMetadataEvidenceState=", validator, StringComparison.Ordinal);
        Assert.Contains("dependency-runtime-probe-only", validator, StringComparison.Ordinal);
        Assert.Contains("compile-surface-proof", validator, StringComparison.Ordinal);
        Assert.Contains("-not $evidence.canPublishPublicly", validator, StringComparison.Ordinal);
    }

    private static void AssertManifest(string manifest, string line)
    {
        Assert.Contains($"trt{line}-onnx-parser-get-layer-output-tensor-metadata", manifest, StringComparison.Ordinal);
        Assert.Contains($"jyppx_trt{line}_onnx_parser_get_layer_output_tensor_metadata", manifest, StringComparison.Ordinal);
        Assert.Contains("\"ownership\": \"caller-owned\"", manifest, StringComparison.Ordinal);
        Assert.Contains($"JYPPX_TENSORRT_VERSION_MAJOR_NUM == {line}", manifest, StringComparison.Ordinal);
        Assert.Contains("\"type\": \"char*\", \"direction\": \"out\", \"managedType\": \"byte[]\"", manifest, StringComparison.Ordinal);
        Assert.Contains("\"moduleManagedType\": \"out NativeTensorRtDims64\"", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX_TensorRtTensor*", manifest, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
