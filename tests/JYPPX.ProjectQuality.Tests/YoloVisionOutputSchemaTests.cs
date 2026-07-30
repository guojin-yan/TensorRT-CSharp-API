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

    [Fact]
    public void SegmentationMaskArtifactSchemaAndIndependentReferenceScriptAreStrict()
    {
        string schemaPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "YoloVision",
            "yolovision-segmentation-mask-artifacts.schema.json");
        string scriptPath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-YoloVisionSegmentationReference.py");
        Assert.True(File.Exists(schemaPath), schemaPath);
        Assert.True(File.Exists(scriptPath), scriptPath);

        string schema = File.ReadAllText(schemaPath);
        using JsonDocument document = JsonDocument.Parse(schema);
        JsonElement root = document.RootElement;
        Assert.Equal(
            "yolovision-segmentation-mask-artifacts.v1",
            root.GetProperty("properties").GetProperty("schemaVersion").GetProperty("const").GetString());
        foreach (string term in new[]
        {
            "prototype-grid-probability",
            "source-image-probability",
            "source-image-thresholded",
            "float32-little-endian",
            "uint8-0-or-1",
            "sha256",
            "isRuntimeProof",
            "isPackageConsumerProof",
            "isPostPublishProof"
        })
        {
            Assert.Contains(term, schema, StringComparison.Ordinal);
        }

        string script = File.ReadAllText(scriptPath);
        foreach (string term in new[]
        {
            "independent-ultralytics-pytorch-cpu-reference",
            "retina_masks=True",
            "yolovision-segmentation-mask-artifacts.v1",
            "spatialTransformApplied",
            "Thresholded mask SHA256 does not match the manifest.",
            "Thresholded mask values must be exactly 0 or 1.",
            "Thresholded mask activePixelCount does not match the file.",
            "minimum_mask_iou",
            "ComparisonPassed=",
            "return 0 if comparison[\"passed\"] else 2",
            "not Owner acceptance",
            "package-consumer proof",
            "post-publish proof"
        })
        {
            Assert.Contains(term, script, StringComparison.Ordinal);
        }
    }
}
