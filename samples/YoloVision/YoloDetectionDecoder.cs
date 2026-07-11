using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public static class YoloDetectionDecoder
{
    public static IReadOnlyList<YoloDetection> Decode(float[] values, int[] dims, YoloPostprocessOptions options)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        YoloOutputLayout resolvedLayout = YoloOutputLayoutInference.InferRank3(dims, options.Layout);
        bool channelsFirst = resolvedLayout == YoloOutputLayout.ChannelsFirst;
        int channelCount = channelsFirst ? dims[1] : dims[2];
        int boxCount = channelsFirst ? dims[2] : dims[1];
        if (values.Length != checked(channelCount * boxCount))
        {
            throw new ArgumentException("Output value count does not match the output tensor shape.", nameof(values));
        }

        bool hasObjectness = ResolveHasObjectness(channelCount, options);
        int classOffset = hasObjectness ? 5 : 4;
        int classCount = options.ClassCount > 0 ? options.ClassCount : channelCount - classOffset;
        if (classCount <= 0 || classOffset + classCount > channelCount)
        {
            throw new NotSupportedException($"YOLO output channel count {channelCount} does not leave room for {classCount} class scores.");
        }

        List<YoloDetection> candidates = new List<YoloDetection>();
        for (int box = 0; box < boxCount; box++)
        {
            float objectness = hasObjectness ? Read(values, channelsFirst, channelCount, boxCount, box, 4) : 1.0f;
            int bestClass = 0;
            float bestClassScore = float.NegativeInfinity;
            for (int classIndex = 0; classIndex < classCount; classIndex++)
            {
                float classScore = Read(values, channelsFirst, channelCount, boxCount, box, classOffset + classIndex);
                if (classScore > bestClassScore)
                {
                    bestClassScore = classScore;
                    bestClass = classIndex;
                }
            }

            float score = objectness * bestClassScore;
            if (score >= options.ConfidenceThreshold)
            {
                candidates.Add(new YoloDetection(
                    bestClass,
                    score,
                    Read(values, channelsFirst, channelCount, boxCount, box, 0),
                    Read(values, channelsFirst, channelCount, boxCount, box, 1),
                    Read(values, channelsFirst, channelCount, boxCount, box, 2),
                    Read(values, channelsFirst, channelCount, boxCount, box, 3),
                    box));
            }
        }

        IReadOnlyList<YoloDetection> ranked = candidates
            .OrderByDescending(static item => item.Score)
            .ToArray();

        if (options.ApplyNms)
        {
            ranked = options.NmsMode == YoloNmsMode.ClassAgnostic
                ? ApplyClassAgnosticNms(ranked, options.IouThreshold)
                : ApplyClassAwareNms(ranked, options.IouThreshold);
        }

        return ranked
            .OrderByDescending(static item => item.Score)
            .Take(options.TopK)
            .ToArray();
    }

    public static IReadOnlyList<YoloDetection> ApplyClassAwareNms(IEnumerable<YoloDetection> detections, float iouThreshold)
    {
        return ApplyNms(detections, iouThreshold, classAware: true);
    }

    public static IReadOnlyList<YoloDetection> ApplyClassAgnosticNms(IEnumerable<YoloDetection> detections, float iouThreshold)
    {
        return ApplyNms(detections, iouThreshold, classAware: false);
    }

    private static IReadOnlyList<YoloDetection> ApplyNms(IEnumerable<YoloDetection> detections, float iouThreshold, bool classAware)
    {
        if (detections == null)
        {
            throw new ArgumentNullException(nameof(detections));
        }

        List<YoloDetection> kept = new List<YoloDetection>();
        foreach (YoloDetection candidate in detections.OrderByDescending(static item => item.Score))
        {
            bool suppressed = false;
            foreach (YoloDetection accepted in kept)
            {
                if ((!classAware || candidate.ClassIndex == accepted.ClassIndex) && candidate.IntersectionOverUnion(accepted) > iouThreshold)
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

    private static bool ResolveHasObjectness(int channelCount, YoloPostprocessOptions options)
    {
        if (options.HasObjectness.HasValue)
        {
            return options.HasObjectness.Value;
        }

        if (options.ClassCount > 0)
        {
            if (channelCount == options.ClassCount + 5)
            {
                return true;
            }

            if (channelCount == options.ClassCount + 4)
            {
                return false;
            }
        }

        return channelCount == 85;
    }

    private static float Read(float[] values, bool channelsFirst, int channelCount, int boxCount, int box, int channel)
    {
        int index = channelsFirst
            ? channel * boxCount + box
            : box * channelCount + channel;
        return values[index];
    }
}
