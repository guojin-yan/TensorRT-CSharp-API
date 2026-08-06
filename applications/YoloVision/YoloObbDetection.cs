namespace YoloVisionSample;

public readonly struct YoloObbDetection
{
    public YoloObbDetection(YoloDetection box, float angleRadians)
    {
        Box = box;
        AngleRadians = angleRadians;
    }

    public YoloDetection Box { get; }

    public float AngleRadians { get; }
}
