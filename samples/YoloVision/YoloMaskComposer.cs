using System;

namespace YoloVisionSample;

public static class YoloMaskComposer
{
    public static YoloSegmentationMask ComposeLinearMask(float[] coefficients, float[] prototypes, int prototypeCount, int width, int height)
    {
        if (coefficients == null)
        {
            throw new ArgumentNullException(nameof(coefficients));
        }

        if (prototypes == null)
        {
            throw new ArgumentNullException(nameof(prototypes));
        }

        if (prototypeCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(prototypeCount));
        }

        if (coefficients.Length != prototypeCount)
        {
            throw new ArgumentException("Coefficient count must match prototype count.", nameof(coefficients));
        }

        int planeSize = checked(width * height);
        if (prototypes.Length != checked(prototypeCount * planeSize))
        {
            throw new ArgumentException("Prototype value count must match prototypeCount * width * height.", nameof(prototypes));
        }

        float[] values = new float[planeSize];
        for (int prototype = 0; prototype < prototypeCount; prototype++)
        {
            int offset = prototype * planeSize;
            float coefficient = coefficients[prototype];
            for (int index = 0; index < planeSize; index++)
            {
                values[index] += coefficient * prototypes[offset + index];
            }
        }

        return new YoloSegmentationMask(
            width,
            height,
            values,
            YoloSegmentationMaskValueKind.RawLogits,
            YoloSegmentationMask.DefaultThreshold);
    }

    public static YoloSegmentationMask ComposeProbabilityMask(
        float[] coefficients,
        float[] prototypes,
        int prototypeCount,
        int width,
        int height,
        float threshold = YoloSegmentationMask.DefaultThreshold)
    {
        YoloSegmentationMask logits = ComposeLinearMask(coefficients, prototypes, prototypeCount, width, height);
        float[] probabilities = new float[logits.Values.Length];
        for (int index = 0; index < logits.Values.Length; index++)
        {
            probabilities[index] = Sigmoid(logits.Values[index]);
        }

        return new YoloSegmentationMask(
            width,
            height,
            probabilities,
            YoloSegmentationMaskValueKind.Probability,
            threshold);
    }

    public static float Sigmoid(float value)
    {
        if (value >= 0.0f)
        {
            return 1.0f / (1.0f + MathF.Exp(-value));
        }

        float exponential = MathF.Exp(value);
        return exponential / (1.0f + exponential);
    }
}
