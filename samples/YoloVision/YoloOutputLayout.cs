namespace YoloVisionSample;

public enum YoloOutputLayout
{
    Auto = 0,
    ChannelsFirst,
    BoxesFirst,
    AnchorBased,
    AnchorFree,
    EndToEndNms,
    SemanticMap
}
