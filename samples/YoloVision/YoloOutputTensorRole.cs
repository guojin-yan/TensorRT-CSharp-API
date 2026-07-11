namespace YoloVisionSample;

public enum YoloOutputTensorRole
{
    Detection = 0,
    Classification,
    SemanticMap,
    MaskPrototypes,
    MaskCoefficients,
    ObbAngles,
    PoseKeypoints
}
