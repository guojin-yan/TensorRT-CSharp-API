using System;

namespace YoloVisionSample;

public static class YoloOutputLayoutInference
{
    public static YoloOutputLayout Parse(string value)
    {
        string normalized = (value ?? string.Empty).Trim().Replace("-", string.Empty, StringComparison.Ordinal).Replace("_", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
        return normalized switch
        {
            "" or "auto" => YoloOutputLayout.Auto,
            "channelsfirst" or "chw" or "bcn" => YoloOutputLayout.ChannelsFirst,
            "boxesfirst" or "nhwc" or "bnc" => YoloOutputLayout.BoxesFirst,
            "anchorbased" => YoloOutputLayout.AnchorBased,
            "anchorfree" => YoloOutputLayout.AnchorFree,
            "end2end" or "endtoend" or "endtoendnms" or "nms" => YoloOutputLayout.EndToEndNms,
            "semantic" or "semanticmap" => YoloOutputLayout.SemanticMap,
            _ => throw new ArgumentException($"Unsupported YOLO output layout '{value}'.")
        };
    }

    public static YoloOutputLayout InferRank3(int[] dims, YoloOutputLayout requestedLayout)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        if (dims.Length != 3 || dims[0] != 1)
        {
            throw new NotSupportedException($"Expected a rank-3 YOLO output such as [1, 84, 8400] or [1, 8400, 84], got [{string.Join(", ", dims)}].");
        }

        if (requestedLayout == YoloOutputLayout.ChannelsFirst || requestedLayout == YoloOutputLayout.AnchorBased || requestedLayout == YoloOutputLayout.AnchorFree)
        {
            return YoloOutputLayout.ChannelsFirst;
        }

        if (requestedLayout == YoloOutputLayout.BoxesFirst || requestedLayout == YoloOutputLayout.EndToEndNms)
        {
            return YoloOutputLayout.BoxesFirst;
        }

        if (requestedLayout != YoloOutputLayout.Auto)
        {
            return requestedLayout;
        }

        return dims[2] > dims[1] ? YoloOutputLayout.ChannelsFirst : YoloOutputLayout.BoxesFirst;
    }
}
