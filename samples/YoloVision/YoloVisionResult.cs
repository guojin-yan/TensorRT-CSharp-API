using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public sealed class YoloVisionResult
{
    private static readonly IReadOnlyList<YoloDetection> EmptyDetections = Array.Empty<YoloDetection>();
    private static readonly IReadOnlyList<YoloClassificationPrediction> EmptyClassifications = Array.Empty<YoloClassificationPrediction>();
    private static readonly IReadOnlyList<YoloSegmentationPrediction> EmptySegmentations = Array.Empty<YoloSegmentationPrediction>();
    private static readonly IReadOnlyList<YoloObbDetection> EmptyOrientedBoxes = Array.Empty<YoloObbDetection>();
    private static readonly IReadOnlyList<YoloPosePrediction> EmptyPoses = Array.Empty<YoloPosePrediction>();

    private YoloVisionResult(
        YoloTaskType taskType,
        IReadOnlyList<YoloDetection> detections,
        IReadOnlyList<YoloClassificationPrediction> classifications,
        IReadOnlyList<YoloSegmentationPrediction> segmentations,
        IReadOnlyList<YoloObbDetection> orientedBoxes,
        IReadOnlyList<YoloPosePrediction> poses,
        YoloSemanticMap? semanticMap,
        string diagnostic)
    {
        TaskType = taskType;
        Detections = detections ?? EmptyDetections;
        Classifications = classifications ?? EmptyClassifications;
        Segmentations = segmentations ?? EmptySegmentations;
        OrientedBoxes = orientedBoxes ?? EmptyOrientedBoxes;
        Poses = poses ?? EmptyPoses;
        SemanticMap = semanticMap;
        Diagnostic = diagnostic ?? string.Empty;
    }

    public YoloTaskType TaskType { get; }

    public IReadOnlyList<YoloDetection> Detections { get; }

    public IReadOnlyList<YoloClassificationPrediction> Classifications { get; }

    public IReadOnlyList<YoloSegmentationPrediction> Segmentations { get; }

    public IReadOnlyList<YoloObbDetection> OrientedBoxes { get; }

    public IReadOnlyList<YoloPosePrediction> Poses { get; }

    public YoloSemanticMap? SemanticMap { get; }

    public string Diagnostic { get; }

    public bool HasSemanticMap => SemanticMap != null;

    public bool HasSegmentationMasks => Segmentations.Count > 0;

    public bool HasOrientedBoxes => OrientedBoxes.Count > 0;

    public bool HasPoses => Poses.Count > 0;

    public bool HasDiagnostic => !string.IsNullOrWhiteSpace(Diagnostic);

    public static YoloVisionResult FromDetections(YoloTaskType taskType, IReadOnlyList<YoloDetection> detections, string diagnostic = "")
    {
        return new YoloVisionResult(taskType, detections, EmptyClassifications, EmptySegmentations, EmptyOrientedBoxes, EmptyPoses, null, diagnostic);
    }

    public static YoloVisionResult FromClassifications(IReadOnlyList<YoloClassificationPrediction> classifications, string diagnostic = "")
    {
        return new YoloVisionResult(YoloTaskType.Classification, EmptyDetections, classifications, EmptySegmentations, EmptyOrientedBoxes, EmptyPoses, null, diagnostic);
    }

    public static YoloVisionResult FromSemanticMap(YoloSemanticMap semanticMap, string diagnostic = "")
    {
        if (semanticMap == null)
        {
            throw new ArgumentNullException(nameof(semanticMap));
        }

        return new YoloVisionResult(YoloTaskType.SemanticSegmentation, EmptyDetections, EmptyClassifications, EmptySegmentations, EmptyOrientedBoxes, EmptyPoses, semanticMap, diagnostic);
    }

    public static YoloVisionResult FromSegmentations(IReadOnlyList<YoloSegmentationPrediction> segmentations, string diagnostic = "")
    {
        IReadOnlyList<YoloSegmentationPrediction> safeSegmentations = segmentations ?? EmptySegmentations;
        IReadOnlyList<YoloDetection> detections = safeSegmentations.Select(static item => item.Detection).ToArray();
        return new YoloVisionResult(YoloTaskType.Segmentation, detections, EmptyClassifications, safeSegmentations, EmptyOrientedBoxes, EmptyPoses, null, diagnostic);
    }

    public static YoloVisionResult FromOrientedBoxes(IReadOnlyList<YoloObbDetection> orientedBoxes, string diagnostic = "")
    {
        IReadOnlyList<YoloObbDetection> safeOrientedBoxes = orientedBoxes ?? EmptyOrientedBoxes;
        IReadOnlyList<YoloDetection> detections = safeOrientedBoxes.Select(static item => item.Box).ToArray();
        return new YoloVisionResult(YoloTaskType.OrientedBoundingBox, detections, EmptyClassifications, EmptySegmentations, safeOrientedBoxes, EmptyPoses, null, diagnostic);
    }

    public static YoloVisionResult FromPoses(IReadOnlyList<YoloPosePrediction> poses, string diagnostic = "")
    {
        IReadOnlyList<YoloPosePrediction> safePoses = poses ?? EmptyPoses;
        IReadOnlyList<YoloDetection> detections = safePoses.Select(static item => item.Detection).ToArray();
        return new YoloVisionResult(YoloTaskType.Pose, detections, EmptyClassifications, EmptySegmentations, EmptyOrientedBoxes, safePoses, null, diagnostic);
    }

    public override string ToString()
    {
        string semantic = SemanticMap == null
            ? "none"
            : $"{SemanticMap.ClassCount}x{SemanticMap.Height}x{SemanticMap.Width}";
        return $"Task={TaskType};Detections={Detections.Count};Classifications={Classifications.Count};Segmentations={Segmentations.Count};OrientedBoxes={OrientedBoxes.Count};Poses={Poses.Count};Semantic={semantic};Diagnostic={Diagnostic}";
    }
}
