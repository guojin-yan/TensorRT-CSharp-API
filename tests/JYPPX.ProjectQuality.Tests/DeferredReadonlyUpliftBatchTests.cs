using System.Text.Json;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredReadonlyUpliftBatchTests
{
    [Fact]
    public void OnnxModelSupportReportExposesPointerFreeSummary()
    {
        TensorRtOnnxModelSupportReport report = new TensorRtOnnxModelSupportReport(
            isSupported: false,
            supportedSubgraphCount: 1,
            unsupportedSubgraphCount: 1,
            new[]
            {
                new TensorRtOnnxSubgraphSupportInfo(0, true, new long[] { 0, 1 }),
                new TensorRtOnnxSubgraphSupportInfo(1, false, new long[] { 2 })
            });

        TensorRtOnnxModelSupportSummary summary = report.ToSummary();

        Assert.False(summary.IsSupported);
        Assert.Equal(2, summary.CopiedSubgraphCount);
        Assert.Equal(1, summary.CopiedSupportedSubgraphCount);
        Assert.Equal(1, summary.CopiedUnsupportedSubgraphCount);
        Assert.Equal(3, summary.CopiedNodeCount);
        Assert.True(summary.CopiedSubgraphCountsMatchReportedCounts);
        Assert.Equal("copied-readonly-summary", summary.RuntimeEvidenceKind);
        Assert.False(summary.IsRuntimeExecutionEvidence);
        Assert.False(summary.IsRuntimeExecutionProof);
        Assert.True(summary.PointerFreeCopiedSummary);
        Assert.False(summary.CanPromoteRuntimeProof);
        Assert.False(summary.CanPromoteReleaseProof);
        Assert.False(summary.CanDeleteDeferredRecord);
        Assert.Contains("CopiedSubgraphs=2", summary.ToString(), StringComparison.Ordinal);

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxModelSupportReport.cs");
        Assert.Contains("public TensorRtOnnxModelSupportSummary ToSummary()", source);
        Assert.Contains("public sealed class TensorRtOnnxModelSupportSummary", source);
        Assert.Contains("does not expose parser-owned tensor pointers", source);
        Assert.Contains("does not promote", source);
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public nint", source);
    }

    [Fact]
    public void CurrentReadonlyUpliftBatchCoversParserRecorderAndPluginMetadataWithoutRuntimeProofPromotion()
    {
        string candidateText = ReadSource("artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        using JsonDocument document = JsonDocument.Parse(candidateText);
        JsonElement groups = document.RootElement.GetProperty("groups");

        JsonElement pluginField = FindCandidate(groups, "plugin-field-metadata-001");
        JsonElement pluginIdentity = FindCandidate(groups, "plugin-creator-identity-002");
        JsonElement pluginV3 = FindCandidate(groups, "plugin-creator-v3-metadata-design-003");
        JsonElement errorSnapshot = FindCandidate(groups, "error-recorder-snapshot-001");
        JsonElement errorInterface = FindCandidate(groups, "error-recorder-interface-info-design-002");

        Assert.Equal("implemented-with-pointer-free-wrapper", pluginField.GetProperty("implementationStatus").GetString());
        Assert.Equal("implemented-with-pointer-free-wrapper", pluginIdentity.GetProperty("implementationStatus").GetString());
        Assert.Equal("implemented-with-pointer-free-wrapper", errorSnapshot.GetProperty("implementationStatus").GetString());
        Assert.Equal("implemented-safe-alternative-design-gate-not-runtime-proof", pluginV3.GetProperty("implementationStatus").GetString());
        Assert.Equal("design-gate-ready-with-safe-snapshot-alternative", errorInterface.GetProperty("implementationStatus").GetString());

        string parserSupport = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParser.ModelSupport.cs");
        string parserDiagnostics = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserDiagnosticSnapshot.cs");
        string parserReport = ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxModelSupportReport.cs");
        string errorRecorder = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtErrorRecorderSnapshot.cs");
        string pluginGate = ReadSource("src", "JYPPX.TensorRtSharp", "Plugins", "TensorRtPluginCreatorV3MetadataDesignGate.cs");
        string onnxSmoke = ReadSource("smoke", "OnnxToEngineSmokeRunner", "Program.cs");
        string pluginSmoke = ReadSource("smoke", "PluginRegistryInventorySmokeRunner", "Program.cs");
        string callbackSmoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");

        Assert.Contains("public bool LayerOutputTensorExists", parserSupport);
        Assert.Contains("public TensorRtOnnxParserDiagnosticSnapshot GetDiagnosticSnapshot()", parserSupport);
        Assert.Contains("public TensorRtOnnxParserDiagnosticSummary ToSummary()", parserDiagnostics);
        Assert.Contains("public TensorRtOnnxModelSupportSummary ToSummary()", parserReport);
        Assert.Contains("public TensorRtErrorRecorderSummary ToSummary()", errorRecorder);
        Assert.Contains("public bool CopiedRecordCountMatchesErrorCount", errorRecorder);
        Assert.Contains("public bool CanPromoteRuntimeProof", pluginGate);
        Assert.Contains("public bool DeferredRowsStillRequired", pluginGate);

        Assert.Contains("ParserDiagnosticSummary=", onnxSmoke);
        Assert.Contains("ParserModelSupportSummary=", onnxSmoke);
        Assert.Contains("LayerOutputIdentity=", onnxSmoke);
        Assert.Contains("GetCreatorSummaries", pluginSmoke);
        Assert.Contains("GetFieldSummaries", pluginSmoke);
        Assert.Contains("ErrorRecorderSummary=", callbackSmoke);

        string combined = parserSupport + parserDiagnostics + parserReport + errorRecorder + pluginGate;
        Assert.DoesNotContain("public IntPtr", combined);
        Assert.DoesNotContain("public nint", combined);
        Assert.Contains("not-runtime-proof", candidateText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CopiedReadonlySummariesExposeUniformNonProofBoundaryMarkers()
    {
        TensorRtOnnxParserDiagnosticSummary parserSummary = new TensorRtOnnxParserDiagnosticSnapshot(
            TensorRtApiLine.TensorRt11,
            errorCount: 1,
            new[] { CreateParserDiagnostic() },
            "parser diagnostics",
            new[] { "plugin.dll" },
            identityOperatorSupported: true).ToSummary();

        TensorRtOnnxParserRefitterDiagnosticSummary parserRefitterSummary =
            new TensorRtOnnxParserRefitterDiagnosticSnapshot(
                TensorRtApiLine.TensorRt11,
                errorCount: 1,
                new[] { CreateParserDiagnostic() },
                "parser-refitter diagnostics").ToSummary();
        CudaDeviceGraphMemorySummary deviceGraphMemorySummary =
            new CudaDeviceGraphMemoryInfo(0, 16, 32, 64, 128).ToSummary();

        AssertCopiedSummaryBoundary(parserSummary);
        AssertCopiedSummaryBoundary(parserRefitterSummary);
        AssertCopiedSummaryBoundary(deviceGraphMemorySummary);
        Assert.Equal(4, deviceGraphMemorySummary.CopiedScalarCounterCount);

        string[] sourceFiles =
        {
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserDiagnosticSnapshot.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxParserRefitterDiagnosticSnapshot.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Parsing", "TensorRtOnnxModelSupportReport.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Monitoring", "TensorRtErrorRecorderSnapshot.cs"),
            ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntimeDiagnosticSnapshot.cs"),
            ReadSource("src", "JYPPX.CudaSharp", "Memory", "CudaMemoryRangeAttribute.cs"),
            ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDeviceGraphMemoryInfo.cs"),
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphDiagnosticSnapshot.cs"),
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphExecDiagnosticSnapshot.cs")
        };

        foreach (string source in sourceFiles)
        {
            Assert.Contains("public string RuntimeEvidenceKind => \"copied-readonly-summary\"", source);
            Assert.Contains("public bool IsRuntimeExecutionEvidence => false", source);
            Assert.Contains("public bool IsRuntimeExecutionProof => false", source);
            Assert.Contains("public bool CanPromoteReleaseProof => false", source);
            Assert.Contains("public bool CanPromoteRuntimeProof => false", source);
            Assert.Contains("public bool CanDeleteDeferredRecord => false", source);
            Assert.DoesNotContain("public IntPtr", source);
            Assert.DoesNotContain("public nint", source);
        }
    }

    private static TensorRtOnnxParserDiagnostic CreateParserDiagnostic()
    {
        return new TensorRtOnnxParserDiagnostic(
            index: 0,
            code: 2,
            line: 7,
            node: 3,
            description: "copied parser diagnostic",
            file: "file.onnx",
            functionName: "Parse",
            nodeName: "identity",
            nodeOperator: "Identity",
            localFunctionStack: Array.Empty<string>());
    }

    private static void AssertCopiedSummaryBoundary(dynamic summary)
    {
        Assert.Equal("copied-readonly-summary", (string)summary.RuntimeEvidenceKind);
        Assert.False((bool)summary.IsRuntimeExecutionEvidence);
        Assert.False((bool)summary.IsRuntimeExecutionProof);
        Assert.True((bool)summary.PointerFreeCopiedSummary);
        Assert.False((bool)summary.CanPromoteRuntimeProof);
        Assert.False((bool)summary.CanPromoteReleaseProof);
        Assert.False((bool)summary.CanDeleteDeferredRecord);
    }

    private static JsonElement FindCandidate(JsonElement groups, string candidateId)
    {
        foreach (JsonProperty group in groups.EnumerateObject())
        {
            foreach (JsonElement candidate in group.Value.EnumerateArray())
            {
                if (candidate.GetProperty("candidateId").GetString() == candidateId)
                {
                    return candidate;
                }
            }
        }

        throw new InvalidOperationException("Candidate not found: " + candidateId);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}
