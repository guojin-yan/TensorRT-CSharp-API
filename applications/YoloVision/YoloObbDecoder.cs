using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public static class YoloObbDecoder
{
    private const float ProbabilisticIouEpsilon = 1e-7f;

    public static YoloObbDetection Decode(YoloDetection box, float angle, bool angleInDegrees)
    {
        if (!float.IsFinite(angle))
        {
            throw new InvalidOperationException("OBB angle must be finite.");
        }

        float radians = angleInDegrees ? angle * MathF.PI / 180.0f : angle;
        return new YoloObbDetection(box, radians);
    }

    public static float ProbabilisticIntersectionOverUnion(YoloObbDetection first, YoloObbDetection second)
    {
        // Preserve the FP32 operation order used by Ultralytics 8.4.21 batch_probiou.
        (float a1, float b1, float c1) = GetCovariance(first);
        (float a2, float b2, float c2) = GetCovariance(second);
        float a = a1 + a2;
        float b = b1 + b2;
        float c = c1 + c2;
        float denominator = a * b - c * c;
        float xDifference = second.Box.CenterX - first.Box.CenterX;
        float yDifference = first.Box.CenterY - second.Box.CenterY;

        float t1 = (a * yDifference * yDifference + b * xDifference * xDifference) /
            (denominator + ProbabilisticIouEpsilon) * 0.25f;
        float t2 = c * xDifference * yDifference /
            (denominator + ProbabilisticIouEpsilon) * 0.5f;
        float firstDeterminant = MathF.Max(a1 * b1 - c1 * c1, 0.0f);
        float secondDeterminant = MathF.Max(a2 * b2 - c2 * c2, 0.0f);
        float t3 = MathF.Log(
            denominator /
            (4.0f * MathF.Sqrt(firstDeterminant * secondDeterminant) + ProbabilisticIouEpsilon) +
            ProbabilisticIouEpsilon) * 0.5f;
        float bhattacharyyaDistance = Math.Clamp(t1 + t2 + t3, ProbabilisticIouEpsilon, 100.0f);
        float hellingerDistance = MathF.Sqrt(1.0f - MathF.Exp(-bhattacharyyaDistance) + ProbabilisticIouEpsilon);
        return 1.0f - hellingerDistance;
    }

    public static IReadOnlyList<YoloObbDetection> ApplyFastNms(
        IEnumerable<YoloObbDetection> detections,
        float iouThreshold,
        bool classAware)
    {
        if (detections == null)
        {
            throw new ArgumentNullException(nameof(detections));
        }

        if (!float.IsFinite(iouThreshold) || iouThreshold < 0.0f || iouThreshold > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(iouThreshold), "IoU threshold must be finite and in [0, 1].");
        }

        YoloObbDetection[] ranked = detections
            .OrderByDescending(static item => item.Box.Score)
            .ToArray();
        List<YoloObbDetection> kept = new List<YoloObbDetection>(ranked.Length);
        for (int candidateIndex = 0; candidateIndex < ranked.Length; candidateIndex++)
        {
            YoloObbDetection candidate = ranked[candidateIndex];
            bool suppressed = false;
            for (int higherScoreIndex = 0; higherScoreIndex < candidateIndex; higherScoreIndex++)
            {
                YoloObbDetection higherScore = ranked[higherScoreIndex];
                if ((!classAware || candidate.Box.ClassIndex == higherScore.Box.ClassIndex) &&
                    ProbabilisticIntersectionOverUnion(higherScore, candidate) >= iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed)
            {
                kept.Add(candidate);
            }
        }

        return kept;
    }

    private static (float A, float B, float C) GetCovariance(YoloObbDetection detection)
    {
        float widthVariance = detection.Box.Width * detection.Box.Width / 12.0f;
        float heightVariance = detection.Box.Height * detection.Box.Height / 12.0f;
        float cosine = MathF.Cos(detection.AngleRadians);
        float sine = MathF.Sin(detection.AngleRadians);
        float cosineSquared = cosine * cosine;
        float sineSquared = sine * sine;
        return (
            widthVariance * cosineSquared + heightVariance * sineSquared,
            widthVariance * sineSquared + heightVariance * cosineSquared,
            (widthVariance - heightVariance) * cosine * sine);
    }
}
