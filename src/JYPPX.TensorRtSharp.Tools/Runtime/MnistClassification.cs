using System;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Classification output for the ten MNIST logits.
/// 十个 MNIST logits 的分类结果。
/// </summary>
public sealed class MnistClassification
{
    public MnistClassification(float[] probabilities, int predictedDigit, float confidence)
    {
        Probabilities = probabilities ?? throw new ArgumentNullException(nameof(probabilities));
        PredictedDigit = predictedDigit;
        Confidence = confidence;
    }

    public float[] Probabilities { get; }

    public int PredictedDigit { get; }

    public float Confidence { get; }
}
