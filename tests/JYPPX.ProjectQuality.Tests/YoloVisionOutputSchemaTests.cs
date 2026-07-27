using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionOutputSchemaTests
{
    [Fact]
    public void YoloVisionOutputSchemaExistsAndCoversCoreTasks()
    {
        string path = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "yolovision-output.schema.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("YoloVision output schema", root.GetProperty("title").GetString());
        Assert.Contains("not runtime proof", root.GetProperty("description").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] requiredTerms =
        {
            "schemaVersion",
            "task",
            "modelFamily",
            "labels",
            "input",
            "engine",
            "runtime",
            "outputs",
            "postprocess",
            "predictions",
            "materialization",
            "in-memory-from-onnx",
            "modelSha256",
            "sourceKind",
            "preprocessed-image-tensor",
            "image",
            "preprocessedTensor",
            "layout",
            "colorOrder",
            "letterbox",
            "resizeMode",
            "fillValue",
            "valueSha256",
            "valuePreview",
            "bindingMetadata",
            "copied-pointer-free-TensorRtEngineBindingReport",
            "semanticRole",
            "isReadyForEnqueue",
            "bytesPerComponent",
            "componentsPerElement",
            "effectiveBytesPerComponent",
            "effectiveComponentsPerElement",
            "usesDataTypeSizeFallback",
            "formatDescription",
            "profileMinShape",
            "profileOptShape",
            "profileMaxShape",
            "valueCaptured",
            "diagnostics",
            "classCount",
            "license",
            "det",
            "cls",
            "seg",
            "pose",
            "obb",
            "sem",
            "box",
            "classId",
            "className",
            "score",
            "maskShape",
            "maskPixelCount",
            "maskTotalPixelCount",
            "maskThreshold",
            "maskValueKind",
            "maskPixelCountScope",
            "spatialTransform",
            "segmentationSpatialTransform",
            "coordinateSpace",
            "model-input-pixels",
            "cropToDetection",
            "bilinear",
            "finalMaskShape",
            "finalMaskPixelCount",
            "finalMaskTotalPixelCount",
            "finalMaskThreshold",
            "finalMaskValueKind",
            "finalMaskScope",
            "sourceBox",
            "source-image-after-explicit-preprocess-inverse-and-optional-box-crop",
            "explicit-preprocess-metadata-transform; owner must validate exporter-specific mask alignment",
            "keypoints",
            "x",
            "y",
            "visibility",
            "center",
            "size",
            "angle",
            "angleUnit",
            "angleRange",
            "semanticPrediction",
            "classificationPrediction",
            "runtime proof",
            "build-only",
            "dry-run",
            "template",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "readonly diagnostics",
        };

        foreach (string term in requiredTerms)
        {
            Assert.Contains(term, text, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", text, StringComparison.OrdinalIgnoreCase);
    }
}
