using System;

namespace YoloVisionSample;

public sealed class YoloSegmentationPrediction
{
    public YoloSegmentationPrediction(YoloDetection detection, YoloSegmentationMask mask)
    {
        Detection = detection;
        Mask = mask ?? throw new ArgumentNullException(nameof(mask));
    }

    public YoloDetection Detection { get; }

    public YoloSegmentationMask Mask { get; }
}
