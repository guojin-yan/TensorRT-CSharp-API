using System;

namespace YoloVisionSample;

public static class YoloObbDecoder
{
    public static YoloObbDetection Decode(YoloDetection box, float angle, bool angleInDegrees)
    {
        float radians = angleInDegrees ? angle * MathF.PI / 180.0f : angle;
        return new YoloObbDetection(box, radians);
    }
}
