using System;
using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecLayerInfoArtifactTests
{
    [Fact]
    public void LayerInfoArtifactIsTopLevelCopiedMetadataWithClosedProofBoundary()
    {
        OnnxEngineLayerInfoArtifact artifact = new OnnxEngineLayerInfoArtifact(
            requested: true,
            collected: true,
            source: "Build",
            state: "export-written",
            informationFormat: "Json",
            contentKind: "json-document",
            requestedProfilingVerbosity: "detailed",
            layerCount: 1,
            dumpRequested: false,
            exportRequested: true,
            exportWritten: true,
            exportPath: "layer-info.json",
            lengthBytes: 42,
            sha256: new string('a', 64),
            diagnostics: Array.Empty<string>(),
            evidenceBoundary: "copied inspector metadata only");
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: TensorRtApiLine.TensorRt10,
            modelSource: "fixture.onnx",
            enginePath: "fixture.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: true,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly",
            diagnostics: Array.Empty<string>(),
            logLines: Array.Empty<string>(),
            layerInfoArtifact: artifact);

        using JsonDocument document = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement actual = document.RootElement.GetProperty("LayerInfoArtifact");
        Assert.Equal("export-written", actual.GetProperty("State").GetString());
        Assert.Equal("Json", actual.GetProperty("InformationFormat").GetString());
        Assert.Equal("json-document", actual.GetProperty("ContentKind").GetString());
        Assert.True(actual.GetProperty("PointerFreeCopiedSnapshot").GetBoolean());
        Assert.False(actual.GetProperty("CanPromoteRuntimeProof").GetBoolean());
        Assert.False(actual.GetProperty("CanPromoteReleaseProof").GetBoolean());
    }
}
