using System;

namespace YoloVisionSample;

public static class YoloXOutputDecoder
{
    private static readonly int[] Strides = { 8, 16, 32 };

    public static float[] TransformRawOutput(
        float[] values,
        int[] outputShape,
        int[] inputShape,
        YoloPostprocessOptions options)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (outputShape == null)
        {
            throw new ArgumentNullException(nameof(outputShape));
        }

        if (inputShape == null)
        {
            throw new ArgumentNullException(nameof(inputShape));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (outputShape.Length != 3 || outputShape[0] != 1)
        {
            throw new NotSupportedException("YOLOX raw output must have rank 3 with batch size 1, for example [1,8400,85].");
        }

        YoloOutputLayout layout = YoloOutputLayoutInference.InferRank3(outputShape, options.Layout);
        if (layout != YoloOutputLayout.BoxesFirst)
        {
            throw new NotSupportedException("YOLOX raw output must use boxes-first layout [1,boxCount,channelCount].");
        }

        if (inputShape.Length != 4 || inputShape[0] != 1 || inputShape[1] != 3)
        {
            throw new NotSupportedException("YOLOX decoding requires an NCHW input shape [1,3,height,width].");
        }

        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        if (inputHeight <= 0 || inputWidth <= 0 || inputHeight % 32 != 0 || inputWidth % 32 != 0)
        {
            throw new NotSupportedException("YOLOX input height and width must be positive multiples of 32.");
        }

        int boxCount = outputShape[1];
        int channelCount = outputShape[2];
        if (channelCount < 6)
        {
            throw new NotSupportedException("YOLOX raw output must contain cx, cy, width, height, objectness, and at least one class score.");
        }

        if (options.HasObjectness == false)
        {
            throw new NotSupportedException("YOLOX raw output always includes an objectness channel.");
        }

        if (options.ClassCount > 0 && channelCount != options.ClassCount + 5)
        {
            throw new NotSupportedException($"YOLOX output channel count {channelCount} does not match class count {options.ClassCount} plus five box/objectness channels.");
        }

        int expectedBoxCount = 0;
        foreach (int stride in Strides)
        {
            expectedBoxCount = checked(expectedBoxCount + (inputHeight / stride) * (inputWidth / stride));
        }

        if (boxCount != expectedBoxCount)
        {
            throw new NotSupportedException($"YOLOX output box count {boxCount} does not match the {expectedBoxCount} grid cells implied by input {inputHeight}x{inputWidth} and strides 8,16,32.");
        }

        if (values.Length != checked(boxCount * channelCount))
        {
            throw new ArgumentException("YOLOX output value count does not match the output tensor shape.", nameof(values));
        }

        float[] decoded = (float[])values.Clone();
        int boxIndex = 0;
        foreach (int stride in Strides)
        {
            int gridHeight = inputHeight / stride;
            int gridWidth = inputWidth / stride;
            for (int gridY = 0; gridY < gridHeight; gridY++)
            {
                for (int gridX = 0; gridX < gridWidth; gridX++)
                {
                    int offset = boxIndex * channelCount;
                    float centerX = (values[offset] + gridX) * stride;
                    float centerY = (values[offset + 1] + gridY) * stride;
                    float width = MathF.Exp(values[offset + 2]) * stride;
                    float height = MathF.Exp(values[offset + 3]) * stride;
                    if (!float.IsFinite(centerX) || !float.IsFinite(centerY) || !float.IsFinite(width) || !float.IsFinite(height))
                    {
                        throw new InvalidOperationException($"YOLOX raw box {boxIndex} decoded to a non-finite coordinate.");
                    }

                    decoded[offset] = centerX;
                    decoded[offset + 1] = centerY;
                    decoded[offset + 2] = width;
                    decoded[offset + 3] = height;
                    boxIndex++;
                }
            }
        }

        return decoded;
    }
}
