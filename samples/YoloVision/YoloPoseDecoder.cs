using System;

namespace YoloVisionSample;

public static class YoloPoseDecoder
{
    public static YoloPoseKeypoint[] DecodeFlatKeypoints(float[] values, int keypointCount, int stride = 3)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (keypointCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(keypointCount));
        }

        if (stride < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(stride));
        }

        if (values.Length < checked(keypointCount * stride))
        {
            throw new ArgumentException("Keypoint tensor is shorter than keypointCount * stride.", nameof(values));
        }

        YoloPoseKeypoint[] keypoints = new YoloPoseKeypoint[keypointCount];
        for (int index = 0; index < keypointCount; index++)
        {
            int offset = index * stride;
            float score = stride >= 3 ? values[offset + 2] : 1.0f;
            keypoints[index] = new YoloPoseKeypoint(values[offset], values[offset + 1], score);
        }

        return keypoints;
    }
}
