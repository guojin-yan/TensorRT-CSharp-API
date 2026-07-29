using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeSerializationOnnxSupportTests
{
    [Fact]
    public void OnnxParserSupportApisUseCountCopyAndDoNotExposeBorrowedTensorPointers()
    {
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-ninth-batch-onnx-parser-support.manifest.json");
        string manifest11 = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-ninth-batch-onnx-parser-support.manifest.json");
        string source = ReadSource("native", "src", "tensorrt", "common", "onnx_parser_support.inc");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Parsing", "NativeBridgeApi.OnnxParserSupport.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelSupport.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserDiagnosticSnapshot.cs");
        string modelSupportReport = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxModelSupportReport.cs");

        AssertOnnxSupportManifest(manifest10, "10");
        AssertOnnxSupportManifest(manifest11, "11");

        Assert.Contains("get_onnx_parser_used_vc_plugin_libraries_with_seh_guard", source);
        Assert.Contains("copy_string_to_buffer(libraries[index]", source);
        Assert.Contains("onnx_parser_supports_model_v2_with_seh_guard", source);
        Assert.Contains("mark_onnx_parser_support_ready(parser_payload, true)", source);
        Assert.Contains("get_onnx_parser_subgraph_count_with_seh_guard", source);
        Assert.Contains("is_onnx_parser_subgraph_supported_with_seh_guard", source);
        Assert.Contains("get_onnx_parser_subgraph_nodes_with_seh_guard", source);
        Assert.Contains("onnx_parser_layer_output_tensor_exists_with_seh_guard", source);
        Assert.Contains("report_vendor_seh_exception", source);
        Assert.DoesNotContain("JYPPX_TensorRtTensor**", manifest10 + manifest11);

        Assert.Contains("GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);", interop);
        Assert.Contains("pinned.Free();", interop);
        Assert.Contains("GetOnnxParserUsedVCPluginLibraries", interop);
        Assert.Contains("ReadParserErrorString", interop);
        Assert.Contains("TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported", interop);

        Assert.Contains("public IReadOnlyList<string> GetUsedVCPluginLibraries()", wrapper);
        Assert.Contains("public TensorRtOnnxParserDiagnosticSnapshot GetDiagnosticSnapshot()", wrapper);
        Assert.Contains("public TensorRtOnnxModelSupportReport CheckModelSupport", wrapper);
        Assert.Contains("public bool LayerOutputTensorExists", wrapper);
        Assert.Contains("public sealed class TensorRtOnnxParserDiagnosticSnapshot", snapshot);
        Assert.Contains("public TensorRtOnnxParserDiagnosticSummary ToSummary()", snapshot);
        Assert.Contains("public sealed class TensorRtOnnxParserDiagnosticSummary", snapshot);
        Assert.Contains("public IReadOnlyList<TensorRtOnnxParserDiagnostic> Diagnostics", snapshot);
        Assert.Contains("public IReadOnlyList<string> UsedVCPluginLibraries", snapshot);
        Assert.Contains("public bool IdentityOperatorSupported", snapshot);
        Assert.Contains("public int CopiedDiagnosticCount", snapshot);
        Assert.Contains("public int DiagnosticSummaryLength", snapshot);
        Assert.Contains("public int UsedVCPluginLibraryCount", snapshot);
        Assert.Contains("does not call TensorRT", snapshot);
        Assert.Contains("does not expose native pointers", snapshot);
        Assert.Contains("does not promote the snapshot to runtime or external-model proof", snapshot);
        Assert.Contains("public TensorRtOnnxModelSupportSummary ToSummary()", modelSupportReport);
        Assert.Contains("public sealed class TensorRtOnnxModelSupportSummary", modelSupportReport);
        Assert.Contains("public string RuntimeEvidenceKind => \"copied-readonly-summary\"", modelSupportReport);
        Assert.Contains("public bool IsRuntimeExecutionEvidence => false", modelSupportReport);
        Assert.Contains("public bool IsRuntimeExecutionProof => false", modelSupportReport);
        Assert.Contains("public bool CanPromoteReleaseProof => false", modelSupportReport);
        Assert.Contains("public bool CanDeleteDeferredRecord => false", modelSupportReport);
        Assert.DoesNotContain("public IntPtr", wrapper + snapshot + modelSupportReport);
        Assert.DoesNotContain("public nint", wrapper + snapshot + modelSupportReport);
    }

    [Fact]
    public void CoverageMatrixClassifiesTrt11LayerOutputTensorPresenceAsImplementedWithDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("\"IParser\",\"getLayerOutputTensor\",\"IParser::getLayerOutputTensor\",\"onnx-parser\",\"implemented-with-deferred-history\"", comparison);
        Assert.Contains("trt11-onnx-parser-layer-output-tensor-exists", comparison);
        Assert.Contains("trt11-parser-get-layer-output-tensor-deferred", comparison);
        Assert.DoesNotContain("JYPPX_TensorRtTensor**", comparison);
    }

    [Fact]
    public void RuntimeSerializationConfigApisHaveTensorRt10And11NativeAndManagedContracts()
    {
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-serialization-controls.manifest.json");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string header11 = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string source = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "runtime_serialization_refit.inc");
        string interop =
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Serialization", "NativeBridgeApi.EngineSerialization.cs") +
            ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Execution", "NativeBridgeApi.ExecutionContextCreation.cs");
        string serializationConfig = ReadSource("src", "JYPPX.TensorRtSharp", "Serialization", "TensorRtSerializationConfig.cs");
        string runtimeConfig = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntimeConfig.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11Serialization.cs");

        Assert.Contains("trt10-engine-create-serialization-config", manifest10);
        Assert.Contains("trt10-engine-serialize-with-config", manifest10);
        Assert.Contains("trt10-engine-create-runtime-config", manifest10);
        Assert.Contains("trt10-runtime-config-get-execution-context-allocation-strategy", manifest10);
        Assert.Contains("trt10-serialization-config-get-flags", manifest10);
        Assert.Contains("trt10-serialization-config-set-flag", manifest10);
        Assert.Contains("trt10-serialization-config-clear-flag", manifest10);
        Assert.Contains("trt10-serialization-config-get-flag", manifest10);

        AssertSerializationHeaderDeclarations(header10, "10");
        AssertSerializationHeaderDeclarations(header11, "11");

        Assert.Contains("engine_payload->createSerializationConfig()", source);
        Assert.Contains("engine_payload->serializeWithConfig(*config_payload)", source);
        Assert.Contains("engine_payload->createRuntimeConfig()", source);
        Assert.Contains("config_payload->getFlags()", source);
        Assert.Contains("config_payload->setFlag(static_cast<nvinfer1::SerializationFlag>(flag))", source);
        Assert.Contains("config_payload->clearFlag(static_cast<nvinfer1::SerializationFlag>(flag))", source);
        Assert.Contains("config_payload->getExecutionContextAllocationStrategy()", source);

        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_serialization_config_get_flags", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_serialization_config_get_flags", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_config_get_execution_context_allocation_strategy", interop);
        Assert.Contains("TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_config_get_execution_context_allocation_strategy", interop);
        Assert.Contains("BridgeStatusCode.NotSupported", interop);

        Assert.Contains("public TensorRtSerializationFlags Flags", serializationConfig);
        Assert.Contains("public bool SetFlag(TensorRtSerializationFlag flag)", serializationConfig);
        Assert.Contains("public bool ClearFlag(TensorRtSerializationFlag flag)", serializationConfig);
        Assert.Contains("public bool GetFlag(TensorRtSerializationFlag flag)", serializationConfig);
        Assert.Contains("public TensorRtSerializationConfigSummary ToSummary()", serializationConfig);
        Assert.Contains("public sealed class TensorRtSerializationConfigSummary", serializationConfig);
        Assert.Contains("public bool PointerFreeCopiedSummary", serializationConfig);
        Assert.Contains("public bool CanPromoteRuntimeProof", serializationConfig);
        Assert.Contains("public bool CanDeleteDeferredRecord", serializationConfig);
        Assert.Contains("public TensorRtExecutionContextAllocationStrategy AllocationStrategy", runtimeConfig);
        Assert.Contains("public TensorRtRuntimeConfigSummary ToSummary()", runtimeConfig);
        Assert.Contains("public sealed class TensorRtRuntimeConfigSummary", runtimeConfig);
        Assert.Contains("public bool PointerFreeCopiedSummary", runtimeConfig);
        Assert.Contains("public bool CanPromoteRuntimeProof", runtimeConfig);
        Assert.Contains("public bool CanDeleteDeferredRecord", runtimeConfig);
        Assert.Contains("public TensorRtSerializationConfig CreateSerializationConfig()", engine);
        Assert.Contains("public TensorRtRuntimeConfig CreateRuntimeConfig()", engine);
        Assert.Contains("public TensorRtExecutionContext CreateExecutionContext(TensorRtRuntimeConfig runtimeConfig)", engine);
        Assert.DoesNotContain("public IntPtr", serializationConfig + runtimeConfig + engine);
        Assert.DoesNotContain("public nint", serializationConfig + runtimeConfig + engine);
    }

    [Fact]
    public void TensorRt10MinimumWeightStreamingBudgetIsLiftedAsLegacyReadOnlyQuery()
    {
        string manifest10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-runtime-serialization-controls.manifest.json");
        string deferred10 = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-twenty-third-batch-deferred-coverage.manifest.json");
        string header10 = ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h");
        string api10 = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.Trt11RuntimeControls.cs");
        string engine = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11RuntimeControls.cs");
        string smoke = ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs");

        Assert.Contains("trt10-engine-get-minimum-weight-streaming-budget", manifest10);
        Assert.Contains("jyppx_trt10_engine_get_minimum_weight_streaming_budget", manifest10);
        Assert.Contains("\"type\": \"int64_t*\", \"direction\": \"out\"", manifest10);
        Assert.Contains("trt10-cuda-engine-get-minimum-weight-streaming-budget-deferred", deferred10);

        Assert.Contains("jyppx_trt10_engine_get_minimum_weight_streaming_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)", header10);
        Assert.Contains("engine_payload->getMinimumWeightStreamingBudget()", api10);
        Assert.Contains("validate_output_pointer(out_budget, \"out_budget\")", api10);

        Assert.Contains("public static long GetEngineMinimumWeightStreamingBudget", interop);
        Assert.Contains("TensorRtApiLine.TensorRt10", interop);
        Assert.Contains("jyppx_trt10_engine_get_minimum_weight_streaming_budget", interop);
        Assert.Contains("public long MinimumWeightStreamingBudgetInBytes", engine);
        Assert.Contains("legacy/deprecated", engine);
        Assert.DoesNotContain("public IntPtr", engine);
        Assert.DoesNotContain("public nint", engine);

        Assert.Contains("MinimumBudget=", smoke);
    }

    [Fact]
    public void OnnxToEngineSmokeReportsRuntimeSerializationAndOnnxSupportSignals()
    {
        string program = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");

        Assert.Contains("ProbeModelSupport(builder, logger, model)", program);
        Assert.Contains("parserDiagnosticSnapshot.UsedVCPluginLibraries", program);
        Assert.Contains("ProbeLayerOutputTensor(parser, \"identity\")", program);
        Assert.Contains("ParserModelSupport", program);
        Assert.Contains("ParserModelSupportSummary=", program);
        Assert.Contains("ParserUsedVCPluginLibraries Count=", program);
        Assert.Contains("ParserDiagnosticSnapshot=", program);
        Assert.Contains("ParserDiagnosticSummary=", program);
        Assert.Contains("TensorRtOnnxModelSupportSummary parserModelSupportSummary = parserModelSupport.ToSummary();", program);
        Assert.Contains("parserModelSupportSummary.RuntimeEvidenceKind", program);
        Assert.Contains("parserModelSupportSummary.CanPromoteReleaseProof", program);
        Assert.Contains("parserModelSupportSummary.CanDeleteDeferredRecord", program);
        Assert.Contains("TensorRtOnnxParserDiagnosticSnapshot parserDiagnosticSnapshot = parser.GetDiagnosticSnapshot();", program);
        Assert.Contains("TensorRtOnnxParserDiagnosticSummary parserDiagnosticSummary = parserDiagnosticSnapshot.ToSummary();", program);
        Assert.Contains("LayerOutputIdentity=", program);
        Assert.Contains("CreateSerializationConfig", ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11Serialization.cs"));
        Assert.Contains("SerializationConfigSummary=", ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs"));
        Assert.Contains("RuntimeConfigSummary=", ReadSource("smoke", "TensorRtSmokeRunner", "Program.cs"));
    }

    private static void AssertOnnxSupportManifest(string manifest, string line)
    {
        Assert.Contains($"trt{line}-onnx-parser-get-used-vc-plugin-library-count", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-used-vc-plugin-library", manifest);
        Assert.Contains($"trt{line}-onnx-parser-supports-model-v2", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-subgraph-count", manifest);
        Assert.Contains($"trt{line}-onnx-parser-is-subgraph-supported", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-subgraph-node-count", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-subgraph-node", manifest);
        Assert.Contains($"trt{line}-onnx-parser-layer-output-tensor-exists", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-supported-subgraph-count", manifest);
        Assert.Contains($"trt{line}-onnx-parser-get-unsupported-subgraph-count", manifest);
        Assert.Contains("\"type\": \"char*\", \"direction\": \"out\", \"managedType\": \"IntPtr\"", manifest);
        Assert.Contains("\"type\": \"const void*\", \"direction\": \"in\", \"managedType\": \"IntPtr\"", manifest);
    }

    private static void AssertSerializationHeaderDeclarations(string header, string line)
    {
        Assert.Contains($"jyppx_trt{line}_engine_create_serialization_config", header);
        Assert.Contains($"jyppx_trt{line}_engine_serialize_with_config", header);
        Assert.Contains($"jyppx_trt{line}_engine_create_runtime_config", header);
        Assert.Contains($"jyppx_trt{line}_serialization_config_get_flags", header);
        Assert.Contains($"jyppx_trt{line}_serialization_config_set_flag", header);
        Assert.Contains($"jyppx_trt{line}_serialization_config_clear_flag", header);
        Assert.Contains($"jyppx_trt{line}_serialization_config_get_flag", header);
        Assert.Contains($"jyppx_trt{line}_runtime_config_get_execution_context_allocation_strategy", header);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
