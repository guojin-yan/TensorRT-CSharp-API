using System;

namespace YoloVisionSample;

public sealed class YoloPostprocessOptions
{
    public YoloPostprocessOptions(
        YoloOutputLayout layout,
        bool? hasObjectness,
        int classCount,
        float confidenceThreshold,
        float iouThreshold,
        int topK,
        bool applyNms)
        : this(layout, hasObjectness, classCount, confidenceThreshold, iouThreshold, topK, applyNms, YoloNmsMode.ClassAware)
    {
    }

    public YoloPostprocessOptions(
        YoloOutputLayout layout,
        bool? hasObjectness,
        int classCount,
        float confidenceThreshold,
        float iouThreshold,
        int topK,
        bool applyNms,
        YoloNmsMode nmsMode)
        : this(
            layout,
            hasObjectness,
            classCount,
            confidenceThreshold,
            iouThreshold,
            topK,
            applyNms,
            nmsMode,
            YoloClassificationScoreMode.Raw)
    {
    }

    public YoloPostprocessOptions(
        YoloOutputLayout layout,
        bool? hasObjectness,
        int classCount,
        float confidenceThreshold,
        float iouThreshold,
        int topK,
        bool applyNms,
        YoloNmsMode nmsMode,
        YoloClassificationScoreMode classificationScoreMode)
    {
        if (classCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(classCount), "Class count must be zero or positive.");
        }

        if (!float.IsFinite(confidenceThreshold) || confidenceThreshold < 0 || confidenceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold), "Confidence threshold must be finite and in [0, 1].");
        }

        if (!float.IsFinite(iouThreshold) || iouThreshold < 0 || iouThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), "IoU threshold must be finite and in [0, 1].");
        }

        if (topK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be positive.");
        }

        if (!Enum.IsDefined(classificationScoreMode))
        {
            throw new ArgumentOutOfRangeException(nameof(classificationScoreMode), "Classification score mode is not defined.");
        }

        if (layout == YoloOutputLayout.EndToEndNms)
        {
            hasObjectness = false;
            applyNms = false;
            nmsMode = YoloNmsMode.None;
        }
        else if (nmsMode == YoloNmsMode.None)
        {
            applyNms = false;
        }

        Layout = layout;
        HasObjectness = hasObjectness;
        ClassCount = classCount;
        ConfidenceThreshold = confidenceThreshold;
        IouThreshold = iouThreshold;
        TopK = topK;
        ApplyNms = applyNms;
        NmsMode = nmsMode;
        ClassificationScoreMode = classificationScoreMode;
    }

    public YoloOutputLayout Layout { get; }

    public bool? HasObjectness { get; }

    public int ClassCount { get; }

    public float ConfidenceThreshold { get; }

    public float IouThreshold { get; }

    public int TopK { get; }

    public bool ApplyNms { get; }

    public YoloNmsMode NmsMode { get; }

    public YoloClassificationScoreMode ClassificationScoreMode { get; }

    public static YoloPostprocessOptions Default { get; } = new YoloPostprocessOptions(
        YoloOutputLayout.Auto,
        hasObjectness: null,
        classCount: 0,
        confidenceThreshold: 0.25f,
        iouThreshold: 0.45f,
        topK: 100,
        applyNms: true);
}
