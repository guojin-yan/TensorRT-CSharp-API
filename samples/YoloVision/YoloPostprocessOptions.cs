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
    {
        if (classCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(classCount), "Class count must be zero or positive.");
        }

        if (confidenceThreshold < 0 || confidenceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold), "Confidence threshold must be in [0, 1].");
        }

        if (iouThreshold < 0 || iouThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), "IoU threshold must be in [0, 1].");
        }

        if (topK <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be positive.");
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
    }

    public YoloOutputLayout Layout { get; }

    public bool? HasObjectness { get; }

    public int ClassCount { get; }

    public float ConfidenceThreshold { get; }

    public float IouThreshold { get; }

    public int TopK { get; }

    public bool ApplyNms { get; }

    public YoloNmsMode NmsMode { get; }

    public static YoloPostprocessOptions Default { get; } = new YoloPostprocessOptions(
        YoloOutputLayout.Auto,
        hasObjectness: null,
        classCount: 0,
        confidenceThreshold: 0.25f,
        iouThreshold: 0.45f,
        topK: 100,
        applyNms: true);
}
