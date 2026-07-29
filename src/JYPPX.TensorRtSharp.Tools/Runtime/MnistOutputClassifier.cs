using System;
using System.Linq;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Stable softmax and argmax helper for MNIST output validation.
/// 用于 MNIST 输出验证的稳定 softmax 与 argmax 帮助器。
/// </summary>
public static class MnistOutputClassifier
{
    public static MnistClassification Classify(float[] logits)
    {
        if (logits == null)
        {
            throw new ArgumentNullException(nameof(logits));
        }

        if (logits.Length != 10)
        {
            throw new ArgumentException("MNIST output must contain exactly ten logits.", nameof(logits));
        }

        float max = logits.Max();
        double[] exponentials = new double[logits.Length];
        double sum = 0.0;
        for (int index = 0; index < logits.Length; index++)
        {
            exponentials[index] = Math.Exp(logits[index] - max);
            sum += exponentials[index];
        }

        float[] probabilities = new float[logits.Length];
        int predictedDigit = 0;
        float confidence = float.MinValue;
        for (int index = 0; index < exponentials.Length; index++)
        {
            probabilities[index] = (float)(exponentials[index] / sum);
            if (probabilities[index] > confidence)
            {
                confidence = probabilities[index];
                predictedDigit = index;
            }
        }

        return new MnistClassification(probabilities, predictedDigit, confidence);
    }
}
