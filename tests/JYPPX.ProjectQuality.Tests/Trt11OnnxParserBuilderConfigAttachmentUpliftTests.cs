using System;
using System.IO;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11OnnxParserBuilderConfigAttachmentUpliftTests
{
    [Fact]
    public void ManifestAndHeaderDeclareOneOwnerSafeTrt11EntryAndKeepDeferredHistory()
    {
        string manifest = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-onnx-parser-builder-config-attachment.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h");
        string deferred = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Equal(1, CountOccurrences(manifest, "\"entryPoint\""));
        Assert.Contains("trt11-onnx-parser-set-builder-config-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TensorRtOnnxParser*", manifest, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TensorRtBuilderConfig*", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", manifest, StringComparison.Ordinal);
        Assert.DoesNotContain("void*", manifest, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_onnx_parser_set_builder_config_safe", header, StringComparison.Ordinal);
        Assert.Contains("trt11-parser-set-builder-config-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeEntryValidatesBothOwnersAndContainsCppAndSehFailures()
    {
        string source = ReadSource("native", "src", "tensorrt", "v11", "modules", "deployment", "onnx_parser_builder_config_attachment.inc");
        string api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");

        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER", source, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG", source, StringComparison.Ordinal);
        Assert.Contains("get_payload<nvonnxparser::IParser>", source, StringComparison.Ordinal);
        Assert.Contains("get_payload<nvinfer1::IBuilderConfig>", source, StringComparison.Ordinal);
        Assert.Contains("parser->setBuilderConfig(config)", source, StringComparison.Ordinal);
        Assert.Contains("capture_vendor_seh_exception_code", source, StringComparison.Ordinal);
        Assert.Contains("report_vendor_seh_exception", source, StringComparison.Ordinal);
        Assert.Contains("report_vendor_exception", source, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11", source, StringComparison.Ordinal);
        Assert.Contains("onnx_parser_builder_config_attachment.inc", api, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedAttachmentRetainsAcceptedConfigAndDisposesParserBeforeLease()
    {
        string attachment = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxParser.BuilderConfig.cs");
        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxParser.cs");
        string interop = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.OnnxParserBuilderConfig.cs");

        Assert.Contains("SafeTensorRtObjectHandleLease.Create(config.Handle)", attachment, StringComparison.Ordinal);
        Assert.Contains("if (!NativeBridgeApi.SetOnnxParserBuilderConfig", attachment, StringComparison.Ordinal);
        Assert.Contains("previousLease = _builderConfigLease", attachment, StringComparison.Ordinal);
        Assert.Contains("_builderConfigLease = pendingLease", attachment, StringComparison.Ordinal);
        Assert.Contains("pendingLease = null", attachment, StringComparison.Ordinal);
        Assert.Contains("previousLease?.Dispose()", attachment, StringComparison.Ordinal);
        Assert.Contains("pendingLease?.Dispose()", attachment, StringComparison.Ordinal);
        Assert.Contains("config.Line != Line", attachment, StringComparison.Ordinal);
        Assert.Contains("Line != TensorRtApiLine.TensorRt11", attachment, StringComparison.Ordinal);
        Assert.Contains("jyppx_trt11_onnx_parser_set_builder_config_safe", interop, StringComparison.Ordinal);

        int parserDispose = parser.IndexOf("_handle.Dispose();", StringComparison.Ordinal);
        int leaseDispose = parser.IndexOf("builderConfigLease?.Dispose();", parserDispose, StringComparison.Ordinal);
        Assert.True(parserDispose >= 0 && leaseDispose > parserDispose);
        Assert.DoesNotContain("public IntPtr", attachment + interop, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", attachment + interop, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", attachment + interop, StringComparison.Ordinal);
    }

    [Fact]
    public void Trt11ParserFlagsUseOfficialValuesAndOlderLinesRemainGuarded()
    {
        Assert.Equal(2, (int)TensorRtOnnxParserFlag.ReportCapabilityDla);
        Assert.Equal(3, (int)TensorRtOnnxParserFlag.EnablePluginOverride);
        Assert.Equal(4, (int)TensorRtOnnxParserFlag.AdjustForDla);
        Assert.Equal(1u << 2, (uint)TensorRtOnnxParserFlags.ReportCapabilityDla);
        Assert.Equal(1u << 3, (uint)TensorRtOnnxParserFlags.EnablePluginOverride);
        Assert.Equal(1u << 4, (uint)TensorRtOnnxParserFlags.AdjustForDla);

        string parser = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOnnxParser.cs");
        Assert.Contains("Line != TensorRtApiLine.TensorRt11 && (flags & trt11OnlyFlags) != 0", parser, StringComparison.Ordinal);
        Assert.Contains("Line != TensorRtApiLine.TensorRt11 && flag >= TensorRtOnnxParserFlag.ReportCapabilityDla", parser, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeToolsAndPackageConsumerCompileTheDeploymentWorkflow()
    {
        string smoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");
        string tools = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("parser.SetBuilderConfig(config)", smoke, StringComparison.Ordinal);
        Assert.Contains("ParserBuilderConfig Supported=", smoke, StringComparison.Ordinal);
        Assert.Contains("parser.SetBuilderConfig(config)", tools, StringComparison.Ordinal);
        Assert.Contains("parser.SetFlag(TensorRtOnnxParserFlag.ReportCapabilityDla)", tools, StringComparison.Ordinal);
        Assert.Contains("parser.SetFlag(TensorRtOnnxParserFlag.AdjustForDla)", tools, StringComparison.Ordinal);
        Assert.Contains("config.SetDefaultDeviceType(TensorRtDeviceType.Dla)", tools, StringComparison.Ordinal);
        Assert.Contains("config.SetDlaCore", tools, StringComparison.Ordinal);
        Assert.Contains("parser.SetBuilderConfig(config)", consumer, StringComparison.Ordinal);
        Assert.Contains("onnx-parser-builder-config-owner-lease", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageUsesExplicitRealAndDeferredHistoryAliases()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        Assert.Contains("\"IParser::setBuilderConfig\" = @(\"id:*onnx-parser-set-builder-config-safe\")", script, StringComparison.Ordinal);
        Assert.Contains("\"IParser::setBuilderConfig\" = @(\"id:*parser-set-builder-config-deferred\")", script, StringComparison.Ordinal);

        int matcherStart = script.IndexOf("function Find-MatchedManifestApis", StringComparison.Ordinal);
        int explicitPriorityStart = script.IndexOf("if ($interfaceKey -in @(", matcherStart, StringComparison.Ordinal);
        int heuristicStart = script.IndexOf("foreach ($candidate in $methodCandidates)", explicitPriorityStart, StringComparison.Ordinal);
        Assert.True(matcherStart >= 0 && explicitPriorityStart > matcherStart && heuristicStart > explicitPriorityStart);
        string explicitPriorityBlock = script.Substring(explicitPriorityStart, heuristicStart - explicitPriorityStart);
        Assert.Contains("\"IParser::setBuilderConfig\"", explicitPriorityBlock, StringComparison.Ordinal);
    }

    [Fact]
    public void GeneratedInteropAndCoveragePromoteTheOfficialRow()
    {
        string generated = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Generated", "NativeMethodsTensorRt.Generated.g.cs");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("jyppx_trt11_onnx_parser_set_builder_config_safe", generated, StringComparison.Ordinal);
        Assert.Contains("\"IParser\",\"setBuilderConfig\",\"IParser::setBuilderConfig\",\"onnx-parser\",\"implemented-with-deferred-history\"", comparison, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string value, string marker)
    {
        int count = 0;
        int index = 0;
        while ((index = value.IndexOf(marker, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += marker.Length;
        }

        return count;
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(parts)));
    }
}
