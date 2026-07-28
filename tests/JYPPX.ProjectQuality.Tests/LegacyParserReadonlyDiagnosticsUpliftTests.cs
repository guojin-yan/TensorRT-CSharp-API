using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class LegacyParserReadonlyDiagnosticsUpliftTests
{
    [Fact]
    public void ManifestUsesTrt8CallerOwnedCopyContractsAndKeepsDeferredHistory()
    {
        string manifest = ReadSource(
            "native", "manifests", "tensorrt", "v8", "trt8-legacy-parser-readonly-diagnostics.manifest.json");
        string deferred = ReadSource(
            "native", "manifests", "tensorrt", "v8", "trt8-cross-version-tenth-batch-other-deferred.manifest.json");

        Assert.Contains("trt8-legacy-uff-get-required-version", manifest, StringComparison.Ordinal);
        Assert.Contains("trt8-legacy-caffe-binary-proto-copy", manifest, StringComparison.Ordinal);
        Assert.Contains("\"ownership\": \"caller-owned\"", manifest, StringComparison.Ordinal);
        Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8", manifest, StringComparison.Ordinal);
        Assert.Contains("\"managedType\": \"byte[]\"", manifest, StringComparison.Ordinal);
        Assert.Contains("trt8-uff-parser-get-uff-required-version-major-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-caffe-parser-parse-binary-proto-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("trt8-binary-proto-blob-get-data-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeBoundaryOwnsParsersAndBlobCopiesDataAndResetsAllOutputs()
    {
        string source = ReadSource(
            "native", "src", "tensorrt", "v8", "modules", "deployment", "legacy_parser_readonly_diagnostics.inc");

        Assert.Contains("std::unique_ptr<nvuffparser::IUffParser>", source, StringComparison.Ordinal);
        Assert.Contains("std::unique_ptr<nvcaffeparser1::ICaffeParser>", source, StringComparison.Ordinal);
        Assert.Contains("std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob>", source, StringComparison.Ordinal);
        Assert.Contains("parser->getUffRequiredVersionMajor()", source, StringComparison.Ordinal);
        Assert.Contains("parser->parseBinaryProto(file_path)", source, StringComparison.Ordinal);
        Assert.Contains("blob->getData()", source, StringComparison.Ordinal);
        Assert.Contains("std::memcpy(output_buffer, data, required_size)", source, StringComparison.Ordinal);
        Assert.Contains("reset_legacy_binary_proto_outputs", source, StringComparison.Ordinal);
        Assert.Contains("*out_major = 0;", source, StringComparison.Ordinal);
        Assert.True(
            source.Split(
                "reset_legacy_uff_version_outputs(out_major, out_minor, out_patch);",
                StringSplitOptions.None).Length - 1 >= 4,
            "UFF outputs must be reset at entry and on every native exception path.");
        Assert.Contains("__try", source, StringComparison.Ordinal);
        Assert.Contains("report_vendor_seh_exception", source, StringComparison.Ordinal);
        Assert.Contains("catch (const std::exception& exception)", source, StringComparison.Ordinal);
        Assert.Contains("catch (...)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("shutdownProtobufLibrary(", source, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsPointerFreeVersionGuardedAndReturnsIndependentCopies()
    {
        string interop = ReadSource(
            "src", "JYPPX.TensorRtSharp", "Internal", "Interop", "NativeBridgeApi.LegacyParserDiagnostics.cs");
        string wrapper = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyParserDiagnostics.cs");
        string snapshots = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtLegacyParserDiagnosticSnapshots.cs");

        Assert.Contains("EnsureLegacyParserLine(line)", interop, StringComparison.Ordinal);
        Assert.Contains("HaveSameDimensions(queriedShape, copiedShape)", interop, StringComparison.Ordinal);
        Assert.DoesNotContain("queriedDimensions).ToString()", interop, StringComparison.Ordinal);
        Assert.Contains("BridgeStatusCode.NotSupported", interop, StringComparison.Ordinal);
        Assert.Contains("public static TensorRtLegacyUffRequiredVersionSnapshot GetUffRequiredVersion", wrapper, StringComparison.Ordinal);
        Assert.Contains("public static TensorRtCaffeBinaryProtoSnapshot ReadCaffeBinaryProto", wrapper, StringComparison.Ordinal);
        Assert.Contains("public byte[] Data => (byte[])_data.Clone();", snapshots, StringComparison.Ordinal);
        Assert.Contains("public bool RetainsNativeParser => false", snapshots, StringComparison.Ordinal);
        Assert.Contains("public bool RetainsNativeObject => false", snapshots, StringComparison.Ordinal);
        Assert.Contains("public bool CallsProcessGlobalProtobufShutdown => false", snapshots, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", wrapper + snapshots, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", wrapper + snapshots, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", wrapper + snapshots, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", wrapper + snapshots, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePackageConsumerAndSmokeContainTheCompleteSafeAlternative()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string smoke = ReadSource("smoke", "LegacyParserDiagnosticsSmokeRunner", "Program.cs");

        Assert.Contains("\"IUffParser::getUffRequiredVersionMajor\" = @(\"id:*legacy-uff-get-required-version\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"ICaffeParser::parseBinaryProto\" = @(\"id:*legacy-caffe-binary-proto-copy\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"IBinaryProtoBlob::getData\" = @(\"id:*legacy-caffe-binary-proto-copy\")", coverage, StringComparison.Ordinal);
        Assert.Contains("id:*uff-parser-get-uff-required-version-major-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("id:*caffe-parser-parse-binary-proto-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("id:*binary-proto-blob-get-data-deferred", coverage, StringComparison.Ordinal);
        Assert.Contains("NvCaffeParser.h", coverage, StringComparison.Ordinal);
        Assert.Contains("NvUffParser.h", coverage, StringComparison.Ordinal);
        Assert.Contains("legacy-parser", coverage, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtLegacyParserDiagnostics.GetUffRequiredVersion)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtLegacyParserDiagnostics.ReadCaffeBinaryProto)", consumer, StringComparison.Ordinal);
        Assert.Contains("trt8-legacy-parser-copied-readonly-diagnostics", consumer, StringComparison.Ordinal);
        Assert.Contains("IndependentCopies={independentCopies}", smoke, StringComparison.Ordinal);
        Assert.Contains("NonTrt8Guard=", smoke, StringComparison.Ordinal);
    }

    [Fact]
    public void CandidateAuditAndRuntimeEvidenceKeepTheReleaseBoundaryExplicit()
    {
        using JsonDocument audit = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt8-legacy-parser-readonly-candidate-audit.json"));
        using JsonDocument evidence = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt8-legacy-parser-readonly-runtime-evidence.json"));

        Assert.Equal(6, audit.RootElement.GetProperty("inventory").GetProperty("selectedCandidates").GetInt32());
        Assert.False(audit.RootElement.GetProperty("publicPointerAdded").GetBoolean());
        Assert.False(audit.RootElement.GetProperty("callsProcessGlobalShutdown").GetBoolean());

        JsonElement runtime = evidence.RootElement.GetProperty("tensorRt8");
        Assert.Equal("copied-readonly-runtime-passed", runtime.GetProperty("state").GetString());
        Assert.Equal("0.6.9", runtime.GetProperty("uffRequiredVersion").GetString());
        Assert.Equal(3136, runtime.GetProperty("binaryProto").GetProperty("dataLength").GetInt32());
        Assert.True(runtime.GetProperty("binaryProto").GetProperty("independentManagedCopies").GetBoolean());
        Assert.True(evidence.RootElement.GetProperty("nonTensorRt8GuardPassed").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
    }

    [Fact]
    public void ArticleAndStrictValidatorRecordTheMeasuredRuntimeResult()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "trt8-legacy-parser-copied-readonly-diagnostics.md");
        string toc = ReadSource("docs", "toc.yml");
        string validator = ReadSource("eng", "Test-LegacyParserReadonlyDiagnosticsEvidence.ps1");

        Assert.Contains("0.6.9", article, StringComparison.Ordinal);
        Assert.Contains("DF7D560B482098FAC1C6122C22BD0A54499ED9F8EC3AC6BAE8FC917D3A01774A", article, StringComparison.Ordinal);
        Assert.Contains("trt8-legacy-parser-copied-readonly-diagnostics.md", toc, StringComparison.Ordinal);
        Assert.Contains("LegacyParserReadonlyDiagnosticsEvidenceState=", validator, StringComparison.Ordinal);
        Assert.Contains("copied-readonly-runtime-passed", validator, StringComparison.Ordinal);
        Assert.Contains("-not $evidence.canPublishPublicly", validator, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
