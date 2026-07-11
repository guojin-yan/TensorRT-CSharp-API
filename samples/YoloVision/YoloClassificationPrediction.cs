namespace YoloVisionSample;

public readonly struct YoloClassificationPrediction
{
    public YoloClassificationPrediction(int classIndex, float score)
    {
        ClassIndex = classIndex;
        Score = score;
    }

    public int ClassIndex { get; }

    public float Score { get; }
}
