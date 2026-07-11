namespace YoloVisionSample;

public readonly struct YoloPoseKeypoint
{
    public YoloPoseKeypoint(float x, float y, float score)
    {
        X = x;
        Y = y;
        Score = score;
    }

    public float X { get; }

    public float Y { get; }

    public float Score { get; }
}
