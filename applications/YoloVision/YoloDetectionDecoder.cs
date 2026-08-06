using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public static class YoloDetectionDecoder
{
    public static IReadOnlyList<YoloDetection> Decode(float[] values, int[] dims, YoloPostprocessOptions options)
    {
        return Decode(values, dims, options, allowTrailingAuxiliaryChannels: false);
    }

    internal static IReadOnlyList<YoloDetection> Decode(
        float[] values,
        int[] dims,
        YoloPostprocessOptions options,
        bool allowTrailingAuxiliaryChannels)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (options.Layout == YoloOutputLayout.EndToEndNms)
        {
            return DecodeEndToEnd(values, dims, options);
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
        bool invalidChannelContract = classCount <= 0
            || classOffset + classCount > channelCount
            || (!allowTrailingAuxiliaryChannels && classOffset + classCount != channelCount);
        if (invalidChannelContract)
        {
            throw new NotSupportedException(
                allowTrailingAuxiliaryChannels
                    ? $"YOLO output channel count {channelCount} does not leave room for {classCount} class scores before auxiliary channels."
                    : $"YOLO output channel count {channelCount} must equal {classOffset} box/objectness channels plus exactly {classCount} class scores.");
        }

        List<YoloDetection> candidates = new List<YoloDetection>();
        for (int box = 0; box < boxCount; box++)
        {
            float objectness = hasObjectness ? Read(values, channelsFirst, channelCount, boxCount, box, 4) : 1.0f;
            if (!float.IsFinite(objectness))
            {
                throw new InvalidOperationException($"Detection row {box} has a non-finite objectness value.");
            }

            int bestClass = 0;
            float bestClassScore = float.NegativeInfinity;
            for (int classIndex = 0; classIndex < classCount; classIndex++)
            {
                float classScore = Read(values, channelsFirst, channelCount, boxCount, box, classOffset + classIndex);
                if (!float.IsFinite(classScore))
                {
                    throw new InvalidOperationException($"Detection row {box} class {classIndex} has a non-finite score.");
                }

                if (classScore > bestClassScore)
                {
                    bestClassScore = classScore;
                    bestClass = classIndex;
                }
            }

            float score = objectness * bestClassScore;
            if (!float.IsFinite(score))
            {
                throw new InvalidOperationException($"Detection row {box} has a non-finite computed score.");
            }

            float centerX = Read(values, channelsFirst, channelCount, boxCount, box, 0);
            float centerY = Read(values, channelsFirst, channelCount, boxCount, box, 1);
            float width = Read(values, channelsFirst, channelCount, boxCount, box, 2);
            float height = Read(values, channelsFirst, channelCount, boxCount, box, 3);
            if (!float.IsFinite(centerX) || !float.IsFinite(centerY) || !float.IsFinite(width) || !float.IsFinite(height))
            {
                throw new InvalidOperationException($"Detection row {box} contains a non-finite box value.");
            }

            if (width < 0.0f || height < 0.0f)
            {
                throw new InvalidOperationException($"Detection row {box} width and height must be zero or positive.");
            }

            if (score >= options.ConfidenceThreshold)
            {
                candidates.Add(new YoloDetection(
                    bestClass,
                    score,
                    centerX,
                    centerY,
                    width,
                    height,
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

    public static IReadOnlyList<YoloDetection> DecodeEndToEnd(float[] values, int[] dims, YoloPostprocessOptions options)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        YoloEndToEndOutput output = YoloEndToEndOutput.FromShape(dims);
        if (values.Length != output.ValueCount)
        {
            throw new ArgumentException("End-to-end output value count does not match the tensor shape.", nameof(values));
        }

        List<YoloDetection> detections = new List<YoloDetection>();
        for (int row = 0; row < output.DetectionCount; row++)
        {
            int offset = row * output.ChannelCount;
            float score = values[offset + 4];
            if (!float.IsFinite(score) || score < 0.0f || score > 1.0f)
            {
                throw new InvalidOperationException($"End-to-end detection row {row} has an invalid score {score}.");
            }

            if (score < options.ConfidenceThreshold)
            {
                continue;
            }

            float left = values[offset];
            float top = values[offset + 1];
            float right = values[offset + 2];
            float bottom = values[offset + 3];
            if (!float.IsFinite(left) || !float.IsFinite(top) || !float.IsFinite(right) || !float.IsFinite(bottom))
            {
                throw new InvalidOperationException($"End-to-end detection row {row} contains a non-finite coordinate.");
            }

            float width = right - left;
            float height = bottom - top;
            if (width <= 0.0f || height <= 0.0f)
            {
                throw new InvalidOperationException($"End-to-end detection row {row} must satisfy x2 > x1 and y2 > y1.");
            }

            float classValue = values[offset + 5];
            if (!float.IsFinite(classValue) || classValue < 0.0f || classValue >= int.MaxValue)
            {
                throw new InvalidOperationException($"End-to-end detection row {row} has an invalid class id {classValue}.");
            }

            float roundedClassValue = MathF.Round(classValue);
            if (MathF.Abs(classValue - roundedClassValue) > 0.0001f)
            {
                throw new InvalidOperationException($"End-to-end detection row {row} class id must be an integer, got {classValue}.");
            }

            int classIndex = checked((int)roundedClassValue);
            if (options.ClassCount > 0 && classIndex >= options.ClassCount)
            {
                throw new InvalidOperationException($"End-to-end detection row {row} class id {classIndex} is outside class count {options.ClassCount}.");
            }

            detections.Add(new YoloDetection(
                classIndex,
                score,
                left + width / 2.0f,
                top + height / 2.0f,
                width,
                height,
                row));
        }

        return detections
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
