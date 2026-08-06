using System;

namespace YoloVisionSample;

public sealed class YoloPosePrediction
{
    public YoloPosePrediction(YoloDetection detection, YoloPoseKeypoint[] keypoints)
    {
        Detection = detection;
        Keypoints = keypoints ?? throw new ArgumentNullException(nameof(keypoints));
    }

    public YoloDetection Detection { get; }

    public YoloPoseKeypoint[] Keypoints { get; }
}
