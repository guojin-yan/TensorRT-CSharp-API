using System;

namespace YoloVisionSample;

public sealed class YoloEndToEndOutput
{
    public const int RequiredChannelCount = 6;

    public YoloEndToEndOutput(int batch, int detectionCount, int channelCount)
    {
        if (batch <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batch));
        }

        if (batch != 1)
        {
            throw new NotSupportedException("YoloVision end-to-end decoding currently supports batch size 1 only.");
        }

        if (detectionCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(detectionCount));
        }

        if (channelCount != RequiredChannelCount)
        {
            throw new NotSupportedException("End-to-end YOLO output must use six columns: x1, y1, x2, y2, score, and classId.");
        }

        Batch = batch;
        DetectionCount = detectionCount;
        ChannelCount = channelCount;
    }

    public int Batch { get; }

    public int DetectionCount { get; }

    public int ChannelCount { get; }

    public int ValueCount => checked(Batch * DetectionCount * ChannelCount);

    public static YoloEndToEndOutput FromShape(int[] dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        if (dims.Length != 3)
        {
            throw new NotSupportedException("End-to-end YOLO output is expected to be rank-3, for example [1, 300, 6].");
        }

        return new YoloEndToEndOutput(dims[0], dims[1], dims[2]);
    }
}
