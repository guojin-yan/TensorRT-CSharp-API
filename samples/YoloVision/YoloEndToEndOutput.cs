using System;

namespace YoloVisionSample;

public sealed class YoloEndToEndOutput
{
    public YoloEndToEndOutput(int batch, int detectionCount, int channelCount)
    {
        if (batch <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batch));
        }

        if (detectionCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(detectionCount));
        }

        if (channelCount < 6)
        {
            throw new ArgumentException("End-to-end NMS output must contain at least box, score, and class channels.", nameof(channelCount));
        }

        Batch = batch;
        DetectionCount = detectionCount;
        ChannelCount = channelCount;
    }

    public int Batch { get; }

    public int DetectionCount { get; }

    public int ChannelCount { get; }

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
