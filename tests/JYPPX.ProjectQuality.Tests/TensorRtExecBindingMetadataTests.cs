using System.Text.Json;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecBindingMetadataTests
{
    [Fact]
    public void BindingMetadataCopiesTensorGeometryFormatAndProofBoundary()
    {
        TensorRtEngineBindingReport bindingReport = new TensorRtEngineBindingReport(
            "binding-fixture",
            profileIndex: 0,
            new[]
            {
                CreateBinding(0, "images", TensorRtIOMode.Input, new[] { -1, 3, 640, 640 }),
                CreateBinding(1, "output", TensorRtIOMode.Output, new[] { -1, 84, 8400 })
            },
            readiness: null);

        OnnxEngineBindingMetadata metadata = OnnxEngineBindingMetadata.FromBindingReport(
            bindingReport,
            "contract-fixture");

        Assert.True(metadata.Attempted);
        Assert.True(metadata.Succeeded);
        Assert.Equal(2, metadata.TensorCount);
        Assert.Equal(1, metadata.InputCount);
        Assert.Equal(1, metadata.OutputCount);
        Assert.False(metadata.ContextReadinessAttached);
        Assert.False(metadata.IsReadyForEnqueue);
        Assert.True(metadata.PointerFreeCopiedSnapshot);
        Assert.False(metadata.CanPromoteRuntimeProof);
        Assert.False(metadata.CanPromoteReleaseProof);
        Assert.Equal("copied-pointer-free-TensorRtEngineBindingReport", metadata.EvidenceKind);
        Assert.Contains("not validate tensor semantics", metadata.EvidenceBoundary, StringComparison.Ordinal);

        OnnxEngineBindingTensorMetadata input = metadata.Tensors[0];
        Assert.Equal("images", input.Name);
        Assert.Equal("Input", input.IOMode);
        Assert.Equal("Float", input.DataType);
        Assert.Equal(new[] { -1, 3, 640, 640 }, input.EngineShape);
        Assert.Equal(new[] { 1, 3, 640, 640 }, input.ProfileMinShape);
        Assert.Equal(new[] { 4, 3, 640, 640 }, input.ProfileMaxShape);
        Assert.Equal("Device", input.Location);
        Assert.Equal("Linear", input.Format);
        Assert.Equal(-1, input.VectorizedDimension);
    }

    [Fact]
    public void DiagnosticsJsonExposesTopLevelBindingMetadata()
    {
        OnnxEngineBindingMetadata metadata = OnnxEngineBindingMetadata.FromBindingReport(
            new TensorRtEngineBindingReport(
                "json-fixture",
                profileIndex: 0,
                new[] { CreateBinding(0, "input", TensorRtIOMode.Input, new[] { 1, 4 }) },
                readiness: null),
            "json-contract");
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: TensorRtApiLine.TensorRt10,
            modelSource: "fixture",
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
            bindingMetadata: metadata);

        using JsonDocument document = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement bindingMetadata = document.RootElement.GetProperty("BindingMetadata");
        Assert.Equal("json-contract", bindingMetadata.GetProperty("State").GetString());
        Assert.Equal(1, bindingMetadata.GetProperty("TensorCount").GetInt32());
        Assert.True(bindingMetadata.GetProperty("PointerFreeCopiedSnapshot").GetBoolean());
        Assert.False(bindingMetadata.GetProperty("CanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("input", bindingMetadata.GetProperty("Tensors")[0].GetProperty("Name").GetString());
    }

    private static TensorRtEngineTensorBinding CreateBinding(
        int index,
        string name,
        TensorRtIOMode ioMode,
        IReadOnlyList<int> engineShape)
    {
        int[] min = engineShape.Select(static value => value < 0 ? 1 : value).ToArray();
        int[] opt = engineShape.Select(static value => value < 0 ? 1 : value).ToArray();
        int[] max = engineShape.Select(static value => value < 0 ? 4 : value).ToArray();
        return new TensorRtEngineTensorBinding(
            index,
            name,
            TensorRtDataType.Float,
            ioMode,
            new TensorRtDims(engineShape.ToArray()),
            TensorRtTensorLocation.Device,
            isShapeInferenceIO: false,
            bytesPerComponent: sizeof(float),
            componentsPerElement: 1,
            TensorRtTensorFormat.Linear,
            formatDescription: "Row major linear FP32",
            vectorizedDimension: -1,
            profileIndex: 0,
            profileMinShape: new TensorRtDims(min),
            profileOptShape: new TensorRtDims(opt),
            profileMaxShape: new TensorRtDims(max),
            diagnostics: Array.Empty<string>());
    }
}
