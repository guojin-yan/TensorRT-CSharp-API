using System;

namespace YoloVisionSample;

public readonly struct YoloDetection
{
    public YoloDetection(int classIndex, float score, float centerX, float centerY, float width, float height)
        : this(classIndex, score, centerX, centerY, width, height, -1)
    {
    }

    public YoloDetection(int classIndex, float score, float centerX, float centerY, float width, float height, int sourceIndex)
    {
        ClassIndex = classIndex;
        Score = score;
        CenterX = centerX;
        CenterY = centerY;
        Width = width;
        Height = height;
        SourceIndex = sourceIndex;
    }

    public int ClassIndex { get; }

    public float Score { get; }

    public float CenterX { get; }

    public float CenterY { get; }

    public float Width { get; }

    public float Height { get; }

    public int SourceIndex { get; }

    public float Left => CenterX - Width / 2.0f;

    public float Top => CenterY - Height / 2.0f;

    public float Right => CenterX + Width / 2.0f;

    public float Bottom => CenterY + Height / 2.0f;

    public float IntersectionOverUnion(YoloDetection other)
    {
        float left = Math.Max(Left, other.Left);
        float top = Math.Max(Top, other.Top);
        float right = Math.Min(Right, other.Right);
        float bottom = Math.Min(Bottom, other.Bottom);
        float intersectionWidth = Math.Max(0.0f, right - left);
        float intersectionHeight = Math.Max(0.0f, bottom - top);
        float intersection = intersectionWidth * intersectionHeight;
        float area = Math.Max(0.0f, Width) * Math.Max(0.0f, Height);
        float otherArea = Math.Max(0.0f, other.Width) * Math.Max(0.0f, other.Height);
        float union = area + otherArea - intersection;
        return union <= 0.0f ? 0.0f : intersection / union;
    }
}
