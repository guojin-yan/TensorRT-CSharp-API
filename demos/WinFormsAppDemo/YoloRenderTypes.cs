using OpenCvSharp;

namespace WinFormsAppDemo;

internal sealed class YoloImagePreprocessResult
{
    public required float[] Tensor { get; init; }

    public required float ResizeScale { get; init; }

    public required int XOffset { get; init; }

    public required int YOffset { get; init; }

    public required int InputWidth { get; init; }

    public required int InputHeight { get; init; }
}

internal readonly record struct CandidateTensorInfo(bool ChannelsFirst, int ChannelCount, int CandidateCount)
{
    public override string ToString()
    {
        return $"{(ChannelsFirst ? "channels-first" : "predictions-first")}, channels={ChannelCount}, candidates={CandidateCount}";
    }
}

internal readonly record struct ProtoTensorInfo(bool ChannelsFirst, int ChannelCount, int Height, int Width)
{
    public override string ToString()
    {
        return $"{(ChannelsFirst ? "channels-first" : "channels-last")}, channels={ChannelCount}, height={Height}, width={Width}";
    }
}

internal readonly record struct PoseKeypointData(float X, float Y, float Score);

internal sealed class DetectionCandidate
{
    public required int ClassId { get; init; }

    public required float Score { get; init; }

    public required Rect OriginalBox { get; init; }

    public required Rect2f InputBox { get; init; }

    public float[]? MaskCoefficients { get; init; }

    public PoseKeypointData[]? Keypoints { get; init; }
}

internal sealed class ObbCandidate
{
    public required int ClassId { get; init; }

    public required float Score { get; init; }

    public required RotatedRect Box { get; init; }
}
