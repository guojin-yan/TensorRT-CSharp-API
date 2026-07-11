using System;

namespace YoloVisionSample;

public sealed class YoloMultiOutputMetadata
{
    public YoloMultiOutputMetadata(
        int maskCoefficientCount = 0,
        int poseKeypointCount = 0,
        int poseKeypointStride = 3,
        bool obbAngleInDegrees = false,
        int? auxiliaryChannelStart = null,
        YoloOutputLayout auxiliaryLayout = YoloOutputLayout.Auto)
    {
        if (maskCoefficientCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maskCoefficientCount), "Mask coefficient count must be zero or positive.");
        }

        if (poseKeypointCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(poseKeypointCount), "Pose keypoint count must be zero or positive.");
        }

        if (poseKeypointStride < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(poseKeypointStride), "Pose keypoint stride must be at least two.");
        }

        if (auxiliaryChannelStart.HasValue && auxiliaryChannelStart.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(auxiliaryChannelStart), "Auxiliary channel start must be zero or positive.");
        }

        MaskCoefficientCount = maskCoefficientCount;
        PoseKeypointCount = poseKeypointCount;
        PoseKeypointStride = poseKeypointStride;
        ObbAngleInDegrees = obbAngleInDegrees;
        AuxiliaryChannelStart = auxiliaryChannelStart;
        AuxiliaryLayout = auxiliaryLayout;
    }

    public int MaskCoefficientCount { get; }

    public int PoseKeypointCount { get; }

    public int PoseKeypointStride { get; }

    public bool ObbAngleInDegrees { get; }

    public int? AuxiliaryChannelStart { get; }

    public YoloOutputLayout AuxiliaryLayout { get; }

    public static YoloMultiOutputMetadata ForSegmentation(int maskCoefficientCount, int? auxiliaryChannelStart = null, YoloOutputLayout auxiliaryLayout = YoloOutputLayout.Auto)
    {
        return new YoloMultiOutputMetadata(maskCoefficientCount: maskCoefficientCount, auxiliaryChannelStart: auxiliaryChannelStart, auxiliaryLayout: auxiliaryLayout);
    }

    public static YoloMultiOutputMetadata ForPose(int keypointCount, int keypointStride = 3, int? auxiliaryChannelStart = null, YoloOutputLayout auxiliaryLayout = YoloOutputLayout.Auto)
    {
        return new YoloMultiOutputMetadata(poseKeypointCount: keypointCount, poseKeypointStride: keypointStride, auxiliaryChannelStart: auxiliaryChannelStart, auxiliaryLayout: auxiliaryLayout);
    }

    public static YoloMultiOutputMetadata ForObb(bool angleInDegrees, int? auxiliaryChannelStart = null, YoloOutputLayout auxiliaryLayout = YoloOutputLayout.Auto)
    {
        return new YoloMultiOutputMetadata(obbAngleInDegrees: angleInDegrees, auxiliaryChannelStart: auxiliaryChannelStart, auxiliaryLayout: auxiliaryLayout);
    }
}
