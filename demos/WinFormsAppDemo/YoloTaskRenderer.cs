using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Runtime.InteropServices;
using CvPoint = OpenCvSharp.Point;
using CvSize = OpenCvSharp.Size;

namespace WinFormsAppDemo;

internal static class YoloTaskRenderer
{
    private static readonly (int Start, int End)[] CocoPoseEdges =
    [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ];

    public static YoloImagePreprocessResult Preprocess(Mat image, int inputWidth, int inputHeight)
    {
        using Mat rgb = new();
        Cv2.CvtColor(image, rgb, ColorConversionCodes.BGR2RGB);

        float resizeScale = Math.Min(inputWidth / (float)rgb.Width, inputHeight / (float)rgb.Height);
        int resizedWidth = Math.Max(1, (int)Math.Round(rgb.Width * resizeScale));
        int resizedHeight = Math.Max(1, (int)Math.Round(rgb.Height * resizeScale));
        int xOffset = (inputWidth - resizedWidth) / 2;
        int yOffset = (inputHeight - resizedHeight) / 2;

        using Mat resized = new();
        Cv2.Resize(rgb, resized, new CvSize(resizedWidth, resizedHeight));

        using Mat padded = new(inputHeight, inputWidth, MatType.CV_8UC3, Scalar.Black);
        using (Mat roi = new(padded, new Rect(xOffset, yOffset, resizedWidth, resizedHeight)))
        {
            resized.CopyTo(roi);
        }

        using Mat normalized = new();
        padded.ConvertTo(normalized, MatType.CV_32FC3, 1.0 / 255.0);

        float[] chwData = new float[inputWidth * inputHeight * 3];
        GCHandle handle = default;

        try
        {
            handle = GCHandle.Alloc(chwData, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            int channelSizeInBytes = inputWidth * inputHeight * sizeof(float);

            for (int channel = 0; channel < 3; channel++)
            {
                using Mat channelMat = Mat.FromPixelData(
                    inputHeight,
                    inputWidth,
                    MatType.CV_32FC1,
                    ptr + (channel * channelSizeInBytes));
                Cv2.ExtractChannel(normalized, channelMat, channel);
            }
        }
        finally
        {
            if (handle.IsAllocated)
            {
                handle.Free();
            }
        }

        return new YoloImagePreprocessResult
        {
            Tensor = chwData,
            ResizeScale = resizeScale,
            XOffset = xOffset,
            YOffset = yOffset,
            InputWidth = inputWidth,
            InputHeight = inputHeight
        };
    }

    public static string DescribeModelOutputs(YoloModelProfile profile, IReadOnlyList<TensorBindingMetadata> outputs)
    {
        return profile.TaskType switch
        {
            YoloTaskType.Detect => DescribeDetectOutputs(profile, outputs),
            YoloTaskType.Segment => DescribeSegmentOutputs(profile, outputs),
            YoloTaskType.Pose => DescribePoseOutputs(profile, outputs),
            YoloTaskType.Obb => DescribeObbOutputs(profile, outputs),
            _ => "Unsupported task"
        };
    }

    public static Mat DrawResults(
        Mat image,
        IReadOnlyList<OutputBufferBinding> outputs,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        out int resultCount)
    {
        return profile.TaskType switch
        {
            YoloTaskType.Detect => DrawDetections(image, outputs, profile, preprocessResult, out resultCount),
            YoloTaskType.Segment => DrawSegments(image, outputs, profile, preprocessResult, out resultCount),
            YoloTaskType.Pose => DrawPoses(image, outputs, profile, preprocessResult, out resultCount),
            YoloTaskType.Obb => DrawObbDetections(image, outputs, profile, preprocessResult, out resultCount),
            _ => throw new NotSupportedException($"Unsupported task type: {profile.TaskType}")
        };
    }

    private static string DescribeDetectOutputs(YoloModelProfile profile, IReadOnlyList<TensorBindingMetadata> outputs)
    {
        TensorBindingMetadata mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Dims);
        int classCount = profile.OutputsAlreadyNms
            ? 1
            : Math.Max(0, outputInfo.ChannelCount - 4 - (profile.HasObjectness ? 1 : 0));
        return $"Main output: {mainTensor.Name} [{mainTensor.Dims}] => {outputInfo}, classes={classCount}";
    }

    private static string DescribeSegmentOutputs(YoloModelProfile profile, IReadOnlyList<TensorBindingMetadata> outputs)
    {
        TensorBindingMetadata mainTensor = ResolveCandidateTensor(outputs);
        TensorBindingMetadata protoTensor = ResolveProtoTensor(outputs, mainTensor);
        CandidateTensorInfo mainInfo = DescribeCandidateTensor(mainTensor.Dims);
        ProtoTensorInfo protoInfo = DescribeProtoTensor(protoTensor.Dims);
        int classCount = profile.OutputsAlreadyNms
            ? 1
            : Math.Max(0, mainInfo.ChannelCount - 4 - protoInfo.ChannelCount - (profile.HasObjectness ? 1 : 0));
        return $"Main output: {mainTensor.Name} [{mainTensor.Dims}] => {mainInfo}, proto: {protoTensor.Name} [{protoTensor.Dims}] => {protoInfo}, classes={classCount}";
    }

    private static string DescribePoseOutputs(YoloModelProfile profile, IReadOnlyList<TensorBindingMetadata> outputs)
    {
        TensorBindingMetadata mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Dims);
        int classCount = profile.OutputsAlreadyNms
            ? 1
            : Math.Max(0, outputInfo.ChannelCount - 4 - (profile.HasObjectness ? 1 : 0) - (profile.PoseKeypointCount * 3));
        return $"Main output: {mainTensor.Name} [{mainTensor.Dims}] => {outputInfo}, classes={classCount}, keypoints={profile.PoseKeypointCount}";
    }

    private static string DescribeObbOutputs(YoloModelProfile profile, IReadOnlyList<TensorBindingMetadata> outputs)
    {
        TensorBindingMetadata mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Dims);
        int classCount = profile.OutputsAlreadyNms
            ? 1
            : Math.Max(0, outputInfo.ChannelCount - 5 - (profile.HasObjectness ? 1 : 0));
        return $"Main output: {mainTensor.Name} [{mainTensor.Dims}] => {outputInfo}, classes={classCount}";
    }

    private static Mat DrawDetections(
        Mat image,
        IReadOnlyList<OutputBufferBinding> outputs,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        out int resultCount)
    {
        OutputBufferBinding mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Metadata.Dims);
        List<DetectionCandidate> candidates = profile.OutputsAlreadyNms
            ? ParseNmsReadyAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                profile,
                preprocessResult)
            : ParseAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                Math.Max(0, outputInfo.ChannelCount - 4 - (profile.HasObjectness ? 1 : 0)),
                profile,
                preprocessResult);

        Mat rendered = image.Clone();
        DrawAxisAlignedDetections(rendered, candidates, profile, profile.OutputsAlreadyNms, out resultCount);
        return rendered;
    }

    private static Mat DrawSegments(
        Mat image,
        IReadOnlyList<OutputBufferBinding> outputs,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        out int resultCount)
    {
        OutputBufferBinding mainTensor = ResolveCandidateTensor(outputs);
        OutputBufferBinding protoTensor = ResolveProtoTensor(outputs, mainTensor);

        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Metadata.Dims);
        ProtoTensorInfo protoInfo = DescribeProtoTensor(protoTensor.Metadata.Dims);
        List<DetectionCandidate> candidates = profile.OutputsAlreadyNms
            ? ParseNmsReadyAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                profile,
                preprocessResult,
                protoInfo.ChannelCount)
            : ParseAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                Math.Max(0, outputInfo.ChannelCount - 4 - protoInfo.ChannelCount - (profile.HasObjectness ? 1 : 0)),
                profile,
                preprocessResult,
                protoInfo.ChannelCount);

        List<DetectionCandidate> kept = profile.OutputsAlreadyNms ? candidates : ApplyAxisAlignedNms(candidates, profile);
        Mat rendered = image.Clone();

        foreach (DetectionCandidate candidate in kept)
        {
            if (candidate.MaskCoefficients is not null)
            {
                using Mat mask = BuildSegmentationMask(
                    protoTensor.HostBuffer,
                    protoInfo,
                    candidate.MaskCoefficients,
                    candidate.InputBox,
                    preprocessResult,
                    image.Width,
                    image.Height,
                    profile.MaskThreshold);

                ApplyMaskOverlay(rendered, mask, GetPaletteColor(candidate.ClassId));
            }

            DrawDetectionBox(rendered, candidate.OriginalBox, candidate.ClassId, candidate.Score, GetPaletteColor(candidate.ClassId));
        }

        resultCount = kept.Count;
        return rendered;
    }

    private static Mat DrawPoses(
        Mat image,
        IReadOnlyList<OutputBufferBinding> outputs,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        out int resultCount)
    {
        OutputBufferBinding mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Metadata.Dims);
        List<DetectionCandidate> candidates = profile.OutputsAlreadyNms
            ? ParseNmsReadyAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                profile,
                preprocessResult,
                0,
                profile.PoseKeypointCount)
            : ParseAxisAlignedCandidates(
                image,
                mainTensor.HostBuffer,
                outputInfo,
                Math.Max(0, outputInfo.ChannelCount - 4 - (profile.HasObjectness ? 1 : 0) - (profile.PoseKeypointCount * 3)),
                profile,
                preprocessResult,
                0,
                profile.PoseKeypointCount);

        List<DetectionCandidate> kept = profile.OutputsAlreadyNms ? candidates : ApplyAxisAlignedNms(candidates, profile);
        Mat rendered = image.Clone();

        foreach (DetectionCandidate candidate in kept)
        {
            Scalar color = GetPaletteColor(candidate.ClassId);
            DrawDetectionBox(rendered, candidate.OriginalBox, candidate.ClassId, candidate.Score, color);
            DrawPose(rendered, candidate.Keypoints ?? [], color, profile.KeypointThreshold);
        }

        resultCount = kept.Count;
        return rendered;
    }

    private static Mat DrawObbDetections(
        Mat image,
        IReadOnlyList<OutputBufferBinding> outputs,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        out int resultCount)
    {
        OutputBufferBinding mainTensor = ResolveCandidateTensor(outputs);
        CandidateTensorInfo outputInfo = DescribeCandidateTensor(mainTensor.Metadata.Dims);
        List<ObbCandidate> candidates = profile.OutputsAlreadyNms
            ? ParseNmsReadyObbCandidates(mainTensor.HostBuffer, outputInfo, profile, preprocessResult)
            : ParseObbCandidates(mainTensor.HostBuffer, outputInfo, profile, preprocessResult);

        int[] keptIndices = profile.OutputsAlreadyNms
            ? Enumerable.Range(0, candidates.Count).ToArray()
            : ApplyObbNms(candidates, profile);

        Mat rendered = image.Clone();
        foreach (int keptIndex in keptIndices)
        {
            ObbCandidate candidate = candidates[keptIndex];
            Scalar color = GetPaletteColor(candidate.ClassId);
            Point2f[] points = candidate.Box.Points();

            for (int i = 0; i < points.Length; i++)
            {
                Cv2.Line(rendered, (CvPoint)points[i], (CvPoint)points[(i + 1) % points.Length], color, 2);
            }

            Cv2.PutText(
                rendered,
                $"{candidate.ClassId} - {candidate.Score:F2}",
                new CvPoint(
                    Math.Clamp((int)points[0].X, 0, rendered.Width - 1),
                    Math.Clamp((int)points[0].Y, 0, rendered.Height - 1)),
                HersheyFonts.HersheySimplex,
                0.7,
                color,
                2);
        }

        resultCount = keptIndices.Length;
        return rendered;
    }

    private static List<DetectionCandidate> ParseAxisAlignedCandidates(
        Mat image,
        float[] output,
        CandidateTensorInfo outputInfo,
        int classCount,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        int maskCoefficientCount = 0,
        int poseKeypointCount = 0)
    {
        List<DetectionCandidate> candidates = [];
        int classOffset = 4 + (profile.HasObjectness ? 1 : 0);

        for (int candidateIndex = 0; candidateIndex < outputInfo.CandidateCount; candidateIndex++)
        {
            float objectness = profile.HasObjectness
                ? GetCandidateValue(output, outputInfo, candidateIndex, 4)
                : 1.0f;

            (int classId, float score) = ResolveBestClassScore(
                output,
                outputInfo,
                candidateIndex,
                classOffset,
                classCount,
                objectness);

            if (score < profile.ConfidenceThreshold)
            {
                continue;
            }

            float centerX = GetCandidateValue(output, outputInfo, candidateIndex, 0);
            float centerY = GetCandidateValue(output, outputInfo, candidateIndex, 1);
            float width = GetCandidateValue(output, outputInfo, candidateIndex, 2);
            float height = GetCandidateValue(output, outputInfo, candidateIndex, 3);

            float x1Input = centerX - (0.5f * width);
            float y1Input = centerY - (0.5f * height);
            Rect2f inputBox = ClipRect2f(new Rect2f(x1Input, y1Input, width, height), preprocessResult.InputWidth, preprocessResult.InputHeight);
            Rect originalBox = ClipRect(
                new Rect(
                    (int)Math.Round((x1Input - preprocessResult.XOffset) / preprocessResult.ResizeScale),
                    (int)Math.Round((y1Input - preprocessResult.YOffset) / preprocessResult.ResizeScale),
                    (int)Math.Round(width / preprocessResult.ResizeScale),
                    (int)Math.Round(height / preprocessResult.ResizeScale)),
                image.Width,
                image.Height);

            if (originalBox.Width <= 1 || originalBox.Height <= 1)
            {
                continue;
            }

            float[]? maskCoefficients = null;
            if (maskCoefficientCount > 0)
            {
                maskCoefficients = new float[maskCoefficientCount];
                int maskOffset = classOffset + classCount;
                for (int i = 0; i < maskCoefficientCount; i++)
                {
                    maskCoefficients[i] = GetCandidateValue(output, outputInfo, candidateIndex, maskOffset + i);
                }
            }

            PoseKeypointData[]? keypoints = null;
            if (poseKeypointCount > 0)
            {
                keypoints = new PoseKeypointData[poseKeypointCount];
                int keypointOffset = classOffset + classCount;

                for (int i = 0; i < poseKeypointCount; i++)
                {
                    float x = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3));
                    float y = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3) + 1);
                    float kpScore = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3) + 2);

                    keypoints[i] = new PoseKeypointData(
                        (x - preprocessResult.XOffset) / preprocessResult.ResizeScale,
                        (y - preprocessResult.YOffset) / preprocessResult.ResizeScale,
                        kpScore);
                }
            }

            candidates.Add(new DetectionCandidate
            {
                ClassId = classId,
                Score = score,
                OriginalBox = originalBox,
                InputBox = inputBox,
                MaskCoefficients = maskCoefficients,
                Keypoints = keypoints
            });
        }

        return candidates;
    }

    private static List<DetectionCandidate> ParseNmsReadyAxisAlignedCandidates(
        Mat image,
        float[] output,
        CandidateTensorInfo outputInfo,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult,
        int maskCoefficientCount = 0,
        int poseKeypointCount = 0)
    {
        List<DetectionCandidate> candidates = [];
        int baseFeatureCount = 6;

        for (int candidateIndex = 0; candidateIndex < outputInfo.CandidateCount; candidateIndex++)
        {
            float score = GetCandidateValue(output, outputInfo, candidateIndex, 4);
            if (score < profile.ConfidenceThreshold)
            {
                continue;
            }

            int classId = (int)Math.Round(GetCandidateValue(output, outputInfo, candidateIndex, 5));

            float x1Input = GetCandidateValue(output, outputInfo, candidateIndex, 0);
            float y1Input = GetCandidateValue(output, outputInfo, candidateIndex, 1);
            float x2Input = GetCandidateValue(output, outputInfo, candidateIndex, 2);
            float y2Input = GetCandidateValue(output, outputInfo, candidateIndex, 3);

            float width = Math.Max(0.0f, x2Input - x1Input);
            float height = Math.Max(0.0f, y2Input - y1Input);

            Rect2f inputBox = ClipRect2f(new Rect2f(x1Input, y1Input, width, height), preprocessResult.InputWidth, preprocessResult.InputHeight);
            Rect originalBox = ClipRect(
                new Rect(
                    (int)Math.Round((x1Input - preprocessResult.XOffset) / preprocessResult.ResizeScale),
                    (int)Math.Round((y1Input - preprocessResult.YOffset) / preprocessResult.ResizeScale),
                    (int)Math.Round(width / preprocessResult.ResizeScale),
                    (int)Math.Round(height / preprocessResult.ResizeScale)),
                image.Width,
                image.Height);

            if (originalBox.Width <= 1 || originalBox.Height <= 1)
            {
                continue;
            }

            float[]? maskCoefficients = null;
            if (maskCoefficientCount > 0)
            {
                maskCoefficients = new float[maskCoefficientCount];
                for (int i = 0; i < maskCoefficientCount; i++)
                {
                    maskCoefficients[i] = GetCandidateValue(output, outputInfo, candidateIndex, baseFeatureCount + i);
                }
            }

            PoseKeypointData[]? keypoints = null;
            if (poseKeypointCount > 0)
            {
                keypoints = new PoseKeypointData[poseKeypointCount];
                int keypointOffset = baseFeatureCount + maskCoefficientCount;

                for (int i = 0; i < poseKeypointCount; i++)
                {
                    float x = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3));
                    float y = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3) + 1);
                    float kpScore = GetCandidateValue(output, outputInfo, candidateIndex, keypointOffset + (i * 3) + 2);

                    keypoints[i] = new PoseKeypointData(
                        (x - preprocessResult.XOffset) / preprocessResult.ResizeScale,
                        (y - preprocessResult.YOffset) / preprocessResult.ResizeScale,
                        kpScore);
                }
            }

            candidates.Add(new DetectionCandidate
            {
                ClassId = classId,
                Score = score,
                OriginalBox = originalBox,
                InputBox = inputBox,
                MaskCoefficients = maskCoefficients,
                Keypoints = keypoints
            });
        }

        return candidates;
    }

    private static List<ObbCandidate> ParseObbCandidates(
        float[] output,
        CandidateTensorInfo outputInfo,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult)
    {
        int classCount = Math.Max(0, outputInfo.ChannelCount - 5 - (profile.HasObjectness ? 1 : 0));
        List<ObbCandidate> candidates = [];

        for (int candidateIndex = 0; candidateIndex < outputInfo.CandidateCount; candidateIndex++)
        {
            float objectness = profile.HasObjectness
                ? GetCandidateValue(output, outputInfo, candidateIndex, 4)
                : 1.0f;

            (int classId, float score) = ResolveBestClassScore(
                output,
                outputInfo,
                candidateIndex,
                profile.HasObjectness ? 5 : 4,
                classCount,
                objectness);

            if (score < profile.ConfidenceThreshold)
            {
                continue;
            }

            float centerX = GetCandidateValue(output, outputInfo, candidateIndex, 0);
            float centerY = GetCandidateValue(output, outputInfo, candidateIndex, 1);
            float width = GetCandidateValue(output, outputInfo, candidateIndex, 2);
            float height = GetCandidateValue(output, outputInfo, candidateIndex, 3);
            float angleRadians = GetCandidateValue(output, outputInfo, candidateIndex, (profile.HasObjectness ? 5 : 4) + classCount);

            AddObbCandidate(candidates, classId, score, centerX, centerY, width, height, angleRadians, preprocessResult);
        }

        return candidates;
    }

    private static List<ObbCandidate> ParseNmsReadyObbCandidates(
        float[] output,
        CandidateTensorInfo outputInfo,
        YoloModelProfile profile,
        YoloImagePreprocessResult preprocessResult)
    {
        List<ObbCandidate> candidates = [];

        for (int candidateIndex = 0; candidateIndex < outputInfo.CandidateCount; candidateIndex++)
        {
            float score = GetCandidateValue(output, outputInfo, candidateIndex, 4);
            if (score < profile.ConfidenceThreshold)
            {
                continue;
            }

            int classId = (int)Math.Round(GetCandidateValue(output, outputInfo, candidateIndex, 5));
            float centerX = GetCandidateValue(output, outputInfo, candidateIndex, 0);
            float centerY = GetCandidateValue(output, outputInfo, candidateIndex, 1);
            float width = GetCandidateValue(output, outputInfo, candidateIndex, 2);
            float height = GetCandidateValue(output, outputInfo, candidateIndex, 3);
            float angle = GetCandidateValue(output, outputInfo, candidateIndex, 6);

            AddObbCandidate(candidates, classId, score, centerX, centerY, width, height, angle, preprocessResult);
        }

        return candidates;
    }

    private static void AddObbCandidate(
        List<ObbCandidate> candidates,
        int classId,
        float score,
        float centerX,
        float centerY,
        float width,
        float height,
        float angle,
        YoloImagePreprocessResult preprocessResult)
    {
        float mappedCenterX = (centerX - preprocessResult.XOffset) / preprocessResult.ResizeScale;
        float mappedCenterY = (centerY - preprocessResult.YOffset) / preprocessResult.ResizeScale;
        float mappedWidth = width / preprocessResult.ResizeScale;
        float mappedHeight = height / preprocessResult.ResizeScale;

        if (mappedWidth <= 1.0f || mappedHeight <= 1.0f)
        {
            return;
        }

        float angleDegrees = Math.Abs(angle) <= (float)(2 * Math.PI)
            ? NormalizeAngle(angle * (float)(180.0 / Math.PI))
            : NormalizeAngle(angle);

        candidates.Add(new ObbCandidate
        {
            ClassId = classId,
            Score = score,
            Box = new RotatedRect(
                new Point2f(mappedCenterX, mappedCenterY),
                new Size2f(mappedWidth, mappedHeight),
                angleDegrees)
        });
    }

    private static List<DetectionCandidate> ApplyAxisAlignedNms(List<DetectionCandidate> candidates, YoloModelProfile profile)
    {
        List<Rect> boxes = candidates.Select(x => x.OriginalBox).ToList();
        List<float> confidences = candidates.Select(x => x.Score).ToList();
        CvDnn.NMSBoxes(boxes, confidences, profile.ConfidenceThreshold, profile.NmsThreshold, out int[] keptIndices);
        return keptIndices.Select(index => candidates[index]).ToList();
    }

    private static int[] ApplyObbNms(List<ObbCandidate> candidates, YoloModelProfile profile)
    {
        List<RotatedRect> boxes = candidates.Select(x => x.Box).ToList();
        List<float> confidences = candidates.Select(x => x.Score).ToList();
        CvDnn.NMSBoxes(boxes, confidences, profile.ConfidenceThreshold, profile.NmsThreshold, out int[] keptIndices);
        return keptIndices;
    }

    private static void DrawAxisAlignedDetections(
        Mat rendered,
        List<DetectionCandidate> candidates,
        YoloModelProfile profile,
        bool skipNms,
        out int resultCount)
    {
        List<DetectionCandidate> kept = skipNms ? candidates : ApplyAxisAlignedNms(candidates, profile);
        foreach (DetectionCandidate candidate in kept)
        {
            DrawDetectionBox(rendered, candidate.OriginalBox, candidate.ClassId, candidate.Score, GetPaletteColor(candidate.ClassId));
        }

        resultCount = kept.Count;
    }

    private static void DrawDetectionBox(Mat image, Rect box, int classId, float score, Scalar color)
    {
        Cv2.Rectangle(image, box, color, 2);

        string label = $"{classId} - {score:F2}";
        int baseLine;
        CvSize textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 1, out baseLine);

        int labelTop = Math.Max(0, box.Y - textSize.Height - 6);
        int labelBottom = Math.Min(image.Height - 1, labelTop + textSize.Height + baseLine + 4);
        int labelRight = Math.Min(image.Width - 1, box.X + textSize.Width + 6);

        Cv2.Rectangle(image, new CvPoint(box.X, labelTop), new CvPoint(labelRight, labelBottom), color, -1);
        Cv2.PutText(
            image,
            label,
            new CvPoint(box.X + 3, Math.Min(image.Height - 1, labelBottom - 4)),
            HersheyFonts.HersheySimplex,
            0.6,
            new Scalar(0, 0, 0),
            1);
    }

    private static void DrawPose(Mat image, PoseKeypointData[] keypoints, Scalar color, float threshold)
    {
        foreach (PoseKeypointData keypoint in keypoints)
        {
            if (keypoint.Score < threshold)
            {
                continue;
            }

            Cv2.Circle(image, new CvPoint((int)keypoint.X, (int)keypoint.Y), 3, color, -1);
        }

        if (keypoints.Length < 17)
        {
            return;
        }

        foreach ((int start, int end) in CocoPoseEdges)
        {
            if (start >= keypoints.Length || end >= keypoints.Length)
            {
                continue;
            }

            if (keypoints[start].Score < threshold || keypoints[end].Score < threshold)
            {
                continue;
            }

            Cv2.Line(
                image,
                new CvPoint((int)keypoints[start].X, (int)keypoints[start].Y),
                new CvPoint((int)keypoints[end].X, (int)keypoints[end].Y),
                color,
                2);
        }
    }

    private static Mat BuildSegmentationMask(
        float[] protoTensor,
        ProtoTensorInfo protoInfo,
        float[] maskCoefficients,
        Rect2f inputBox,
        YoloImagePreprocessResult preprocessResult,
        int originalWidth,
        int originalHeight,
        float threshold)
    {
        int protoArea = protoInfo.Height * protoInfo.Width;
        float[] protoMask = new float[protoArea];

        for (int y = 0; y < protoInfo.Height; y++)
        {
            for (int x = 0; x < protoInfo.Width; x++)
            {
                float sum = 0.0f;
                for (int channel = 0; channel < protoInfo.ChannelCount; channel++)
                {
                    sum += maskCoefficients[channel] * GetProtoValue(protoTensor, protoInfo, channel, y, x);
                }

                protoMask[(y * protoInfo.Width) + x] = Sigmoid(sum);
            }
        }

        float scaleX = protoInfo.Width / (float)preprocessResult.InputWidth;
        float scaleY = protoInfo.Height / (float)preprocessResult.InputHeight;

        int cropLeft = Math.Clamp((int)Math.Floor(inputBox.X * scaleX), 0, protoInfo.Width - 1);
        int cropTop = Math.Clamp((int)Math.Floor(inputBox.Y * scaleY), 0, protoInfo.Height - 1);
        int cropRight = Math.Clamp((int)Math.Ceiling((inputBox.X + inputBox.Width) * scaleX), cropLeft + 1, protoInfo.Width);
        int cropBottom = Math.Clamp((int)Math.Ceiling((inputBox.Y + inputBox.Height) * scaleY), cropTop + 1, protoInfo.Height);

        for (int y = 0; y < protoInfo.Height; y++)
        {
            for (int x = 0; x < protoInfo.Width; x++)
            {
                if (x < cropLeft || x >= cropRight || y < cropTop || y >= cropBottom)
                {
                    protoMask[(y * protoInfo.Width) + x] = 0.0f;
                }
            }
        }

        using Mat protoMaskMat = CreateFloatMat(protoMask, protoInfo.Height, protoInfo.Width);
        using Mat inputMask = new();
        Cv2.Resize(protoMaskMat, inputMask, new CvSize(preprocessResult.InputWidth, preprocessResult.InputHeight), 0, 0, InterpolationFlags.Linear);

        int contentWidth = Math.Clamp((int)Math.Round(originalWidth * preprocessResult.ResizeScale), 1, preprocessResult.InputWidth - preprocessResult.XOffset);
        int contentHeight = Math.Clamp((int)Math.Round(originalHeight * preprocessResult.ResizeScale), 1, preprocessResult.InputHeight - preprocessResult.YOffset);
        Rect contentRect = new(preprocessResult.XOffset, preprocessResult.YOffset, contentWidth, contentHeight);

        using Mat contentMask = new(inputMask, contentRect);
        using Mat originalMask = new();
        Cv2.Resize(contentMask, originalMask, new CvSize(originalWidth, originalHeight), 0, 0, InterpolationFlags.Linear);

        using Mat binaryMask = new();
        Cv2.Threshold(originalMask, binaryMask, threshold, 255, ThresholdTypes.Binary);

        Mat result = new();
        binaryMask.ConvertTo(result, MatType.CV_8UC1);
        return result;
    }

    private static void ApplyMaskOverlay(Mat image, Mat mask, Scalar color)
    {
        using Mat colorLayer = image.Clone();
        colorLayer.SetTo(color, mask);
        Cv2.AddWeighted(colorLayer, 0.35, image, 0.65, 0.0, image);
    }

    private static OutputBufferBinding ResolveCandidateTensor(IReadOnlyList<OutputBufferBinding> outputs)
    {
        OutputBufferBinding? best = null;
        int bestRankScore = int.MinValue;
        int maxCandidateCount = -1;

        foreach (OutputBufferBinding output in outputs)
        {
            int[] meaningfulDims = GetMeaningfulDims(output.Metadata.Dims);
            if (meaningfulDims.Length < 2)
            {
                continue;
            }

            CandidateTensorInfo info = DescribeCandidateTensor(output.Metadata.Dims);
            int rankScore = meaningfulDims.Length == 2 ? 1 : 0;
            if (rankScore > bestRankScore || (rankScore == bestRankScore && info.CandidateCount > maxCandidateCount))
            {
                best = output;
                bestRankScore = rankScore;
                maxCandidateCount = info.CandidateCount;
            }
        }

        return best ?? throw new InvalidOperationException("No output tensors available.");
    }

    private static TensorBindingMetadata ResolveCandidateTensor(IReadOnlyList<TensorBindingMetadata> outputs)
    {
        TensorBindingMetadata? best = null;
        int bestRankScore = int.MinValue;
        int maxCandidateCount = -1;

        foreach (TensorBindingMetadata output in outputs)
        {
            int[] meaningfulDims = GetMeaningfulDims(output.Dims);
            if (meaningfulDims.Length < 2)
            {
                continue;
            }

            CandidateTensorInfo info = DescribeCandidateTensor(output.Dims);
            int rankScore = meaningfulDims.Length == 2 ? 1 : 0;
            if (rankScore > bestRankScore || (rankScore == bestRankScore && info.CandidateCount > maxCandidateCount))
            {
                best = output;
                bestRankScore = rankScore;
                maxCandidateCount = info.CandidateCount;
            }
        }

        return best ?? throw new InvalidOperationException("No output tensors available.");
    }

    private static OutputBufferBinding ResolveProtoTensor(IReadOnlyList<OutputBufferBinding> outputs, OutputBufferBinding mainTensor)
    {
        foreach (OutputBufferBinding output in outputs)
        {
            if (ReferenceEquals(output, mainTensor))
            {
                continue;
            }

            if (GetMeaningfulDims(output.Metadata.Dims).Length >= 3)
            {
                return output;
            }
        }

        throw new InvalidOperationException("Segmentation proto output not found.");
    }

    private static TensorBindingMetadata ResolveProtoTensor(IReadOnlyList<TensorBindingMetadata> outputs, TensorBindingMetadata mainTensor)
    {
        foreach (TensorBindingMetadata output in outputs)
        {
            if (ReferenceEquals(output, mainTensor))
            {
                continue;
            }

            if (GetMeaningfulDims(output.Dims).Length >= 3)
            {
                return output;
            }
        }

        throw new InvalidOperationException("Segmentation proto output not found.");
    }

    private static CandidateTensorInfo DescribeCandidateTensor(Dims dims)
    {
        int[] meaningfulDims = GetMeaningfulDims(dims);
        if (meaningfulDims.Length < 2)
        {
            throw new NotSupportedException($"Unsupported candidate dims: {dims}");
        }

        int first = meaningfulDims[^2];
        int second = meaningfulDims[^1];
        bool channelsFirst = first <= second;
        return new CandidateTensorInfo(channelsFirst, channelsFirst ? first : second, channelsFirst ? second : first);
    }

    private static ProtoTensorInfo DescribeProtoTensor(Dims dims)
    {
        int[] meaningfulDims = GetMeaningfulDims(dims);
        if (meaningfulDims.Length < 3)
        {
            throw new NotSupportedException($"Unsupported proto dims: {dims}");
        }

        if (meaningfulDims[0] <= meaningfulDims[^1] && meaningfulDims[0] <= 256)
        {
            return new ProtoTensorInfo(true, meaningfulDims[0], meaningfulDims[1], meaningfulDims[2]);
        }

        return new ProtoTensorInfo(false, meaningfulDims[^1], meaningfulDims[0], meaningfulDims[1]);
    }

    private static int[] GetMeaningfulDims(Dims dims)
    {
        return dims.d
            .Take(dims.nbDims)
            .Where(value => value > 1)
            .Select(value => (int)value)
            .ToArray();
    }

    private static float GetCandidateValue(float[] data, CandidateTensorInfo info, int candidateIndex, int channelIndex)
    {
        return info.ChannelsFirst
            ? data[(channelIndex * info.CandidateCount) + candidateIndex]
            : data[(candidateIndex * info.ChannelCount) + channelIndex];
    }

    private static float GetProtoValue(float[] data, ProtoTensorInfo info, int channelIndex, int y, int x)
    {
        return info.ChannelsFirst
            ? data[(channelIndex * info.Height * info.Width) + (y * info.Width) + x]
            : data[(y * info.Width * info.ChannelCount) + (x * info.ChannelCount) + channelIndex];
    }

    private static (int ClassId, float Score) ResolveBestClassScore(
        float[] output,
        CandidateTensorInfo outputInfo,
        int candidateIndex,
        int classOffset,
        int classCount,
        float objectness)
    {
        if (classCount <= 0)
        {
            return (0, objectness);
        }

        float bestScore = 0.0f;
        int bestClassId = 0;

        for (int classIndex = 0; classIndex < classCount; classIndex++)
        {
            float classScore = GetCandidateValue(output, outputInfo, candidateIndex, classOffset + classIndex);
            float score = objectness * classScore;
            if (score > bestScore)
            {
                bestScore = score;
                bestClassId = classIndex;
            }
        }

        return (bestClassId, bestScore);
    }

    private static Rect ClipRect(Rect rect, int imageWidth, int imageHeight)
    {
        int left = Math.Clamp(rect.X, 0, imageWidth - 1);
        int top = Math.Clamp(rect.Y, 0, imageHeight - 1);
        int right = Math.Clamp(rect.X + rect.Width, 0, imageWidth);
        int bottom = Math.Clamp(rect.Y + rect.Height, 0, imageHeight);
        return new Rect(left, top, Math.Max(0, right - left), Math.Max(0, bottom - top));
    }

    private static Rect2f ClipRect2f(Rect2f rect, int width, int height)
    {
        float left = Math.Clamp(rect.X, 0.0f, width - 1.0f);
        float top = Math.Clamp(rect.Y, 0.0f, height - 1.0f);
        float right = Math.Clamp(rect.X + rect.Width, 0.0f, width);
        float bottom = Math.Clamp(rect.Y + rect.Height, 0.0f, height);
        return new Rect2f(left, top, Math.Max(0.0f, right - left), Math.Max(0.0f, bottom - top));
    }

    private static Mat CreateFloatMat(float[] data, int height, int width)
    {
        Mat mat = new(height, width, MatType.CV_32FC1);
        Marshal.Copy(data, 0, mat.Data, data.Length);
        return mat;
    }

    private static Scalar GetPaletteColor(int classId)
    {
        Scalar[] palette =
        [
            new Scalar(0, 255, 0),
            new Scalar(255, 128, 0),
            new Scalar(255, 0, 0),
            new Scalar(0, 255, 255),
            new Scalar(255, 0, 255),
            new Scalar(0, 128, 255)
        ];

        return palette[Math.Abs(classId) % palette.Length];
    }

    private static float Sigmoid(float value)
    {
        return 1.0f / (1.0f + MathF.Exp(-value));
    }

    private static float NormalizeAngle(float angleDegrees)
    {
        while (angleDegrees >= 90.0f)
        {
            angleDegrees -= 180.0f;
        }

        while (angleDegrees < -90.0f)
        {
            angleDegrees += 180.0f;
        }

        return angleDegrees;
    }
}
