using System;
using System.Collections.Generic;
using System.Linq;

namespace YoloVisionSample;

public static class YoloSampleRunner
{
    private const int BoxChannelCount = 4;

    public static YoloVisionResult DecodeOutput(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        return profile.TaskType switch
        {
            YoloTaskType.Detection => YoloVisionResult.FromDetections(profile.TaskType, DecodeDetections(values, outputShape, profile)),
            YoloTaskType.Classification => YoloVisionResult.FromClassifications(DecodeClassifications(values, outputShape, profile)),
            YoloTaskType.SemanticSegmentation => YoloVisionResult.FromSemanticMap(DecodeSemanticMap(values, outputShape, profile)),
            YoloTaskType.Segmentation => YoloVisionResult.FromDetections(
                profile.TaskType,
                DecodeDetections(values, outputShape, profile),
                "Segmentation mask prototypes are model-specific auxiliary outputs; this single-output runner decodes the box branch and leaves mask composition to YoloMaskComposer."),
            YoloTaskType.OrientedBoundingBox => YoloVisionResult.FromDetections(
                profile.TaskType,
                DecodeDetections(values, outputShape, profile),
                "OBB angle channels are model-specific; this single-output runner decodes the box branch and leaves angle conversion to YoloObbDecoder."),
            YoloTaskType.Pose => YoloVisionResult.FromDetections(
                profile.TaskType,
                DecodeDetections(values, outputShape, profile),
                "Pose keypoint channels are model-specific; this single-output runner decodes the box branch and leaves copied keypoint slices to YoloPoseDecoder."),
            _ => throw new NotSupportedException($"Unsupported YOLO task {profile.TaskType}.")
        };
    }

    public static IReadOnlyList<YoloDetection> DecodeDetections(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (profile.TaskType != YoloTaskType.Detection && profile.TaskType != YoloTaskType.Segmentation && profile.TaskType != YoloTaskType.OrientedBoundingBox && profile.TaskType != YoloTaskType.Pose)
        {
            throw new NotSupportedException($"Task {profile.TaskType} does not use detection-style box decoding.");
        }

        float[] detectionValues = profile.Family == YoloModelFamily.YoloX
            ? YoloXOutputDecoder.TransformRawOutput(values, outputShape, profile.InputShape, profile.Postprocess)
            : values;
        return YoloDetectionDecoder.Decode(detectionValues, outputShape, profile.Postprocess);
    }

    public static YoloVisionResult DecodeRuntimeOutputs(
        YoloRuntimeOutputSet outputs,
        YoloModelProfile profile,
        YoloMultiOutputMetadata? metadata = null)
    {
        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        return profile.TaskType switch
        {
            YoloTaskType.Detection => DecodeSingleRoleOutput(outputs, profile, YoloOutputTensorRole.Detection),
            YoloTaskType.Classification => DecodeSingleRoleOutput(outputs, profile, YoloOutputTensorRole.Classification),
            YoloTaskType.SemanticSegmentation => DecodeSingleRoleOutput(outputs, profile, YoloOutputTensorRole.SemanticMap),
            YoloTaskType.Segmentation => DecodeSegmentationRuntimeOutputs(outputs, profile, metadata),
            YoloTaskType.OrientedBoundingBox => DecodeObbRuntimeOutputs(outputs, profile, metadata),
            YoloTaskType.Pose => DecodePoseRuntimeOutputs(outputs, profile, metadata),
            _ => throw new NotSupportedException($"Unsupported YOLO task {profile.TaskType}.")
        };
    }

    public static YoloVisionResult DecodeSegmentationOutputs(
        float[] boxValues,
        int[] boxShape,
        float[] prototypeValues,
        int[] prototypeShape,
        YoloModelProfile profile,
        YoloMultiOutputMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        if (metadata.MaskCoefficientCount <= 0)
        {
            throw new ArgumentException("Segmentation metadata must declare a positive mask coefficient count.", nameof(metadata));
        }

        YoloModelProfile detectionProfile = CreateDetectionProfileForAuxiliary(profile, boxShape, metadata.MaskCoefficientCount);
        IReadOnlyList<YoloDetection> detections = DecodeDetections(boxValues, boxShape, detectionProfile);
        float[][] coefficients = ReadPerBoxAuxiliaryRows(boxValues, boxShape, detectionProfile.Postprocess, metadata.AuxiliaryChannelStart, metadata.MaskCoefficientCount, metadata.AuxiliaryLayout);
        PrototypeTensor prototypes = PrototypeTensor.FromValues(prototypeValues, prototypeShape);

        List<YoloSegmentationPrediction> segmentations = new List<YoloSegmentationPrediction>();
        foreach (YoloDetection detection in detections)
        {
            if (detection.SourceIndex < 0 || detection.SourceIndex >= coefficients.Length)
            {
                continue;
            }

            YoloSegmentationMask mask = YoloMaskComposer.ComposeProbabilityMask(
                coefficients[detection.SourceIndex],
                prototypes.Values,
                prototypes.PrototypeCount,
                prototypes.Width,
                prototypes.Height,
                metadata.MaskThreshold);
            segmentations.Add(new YoloSegmentationPrediction(detection, mask));
        }

        return YoloVisionResult.FromSegmentations(segmentations, "Decoded segmentation masks from detection rows and prototype tensor.");
    }

    public static YoloVisionResult DecodePoseOutputs(
        float[] boxValues,
        int[] boxShape,
        float[] keypointValues,
        int[] keypointShape,
        YoloModelProfile profile,
        YoloMultiOutputMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        if (metadata.PoseKeypointCount <= 0)
        {
            throw new ArgumentException("Pose metadata must declare a positive keypoint count.", nameof(metadata));
        }

        YoloModelProfile detectionProfile = CreateDetectionProfile(profile);
        IReadOnlyList<YoloDetection> detections = DecodeDetections(boxValues, boxShape, detectionProfile);
        float[][] keypointRows = ReadAuxiliaryRows(keypointValues, keypointShape, metadata.PoseKeypointCount * metadata.PoseKeypointStride, metadata.AuxiliaryLayout);

        List<YoloPosePrediction> poses = new List<YoloPosePrediction>();
        foreach (YoloDetection detection in detections)
        {
            if (detection.SourceIndex < 0 || detection.SourceIndex >= keypointRows.Length)
            {
                continue;
            }

            YoloPoseKeypoint[] keypoints = YoloPoseDecoder.DecodeFlatKeypoints(
                keypointRows[detection.SourceIndex],
                metadata.PoseKeypointCount,
                metadata.PoseKeypointStride);
            poses.Add(new YoloPosePrediction(detection, keypoints));
        }

        return YoloVisionResult.FromPoses(poses, "Decoded pose keypoints from detection rows and keypoint tensor.");
    }

    public static YoloVisionResult DecodeEmbeddedPoseOutput(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile,
        YoloMultiOutputMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        if (metadata.PoseKeypointCount <= 0)
        {
            throw new ArgumentException("Pose metadata must declare a positive keypoint count.", nameof(metadata));
        }

        int auxiliaryWidth = checked(metadata.PoseKeypointCount * metadata.PoseKeypointStride);
        YoloModelProfile detectionProfile = CreateDetectionProfileForAuxiliary(profile, outputShape, auxiliaryWidth);
        int channelStart = ValidateEmbeddedAuxiliaryContract(outputShape, detectionProfile.Postprocess, metadata, auxiliaryWidth);
        IReadOnlyList<YoloDetection> detections = DecodeDetections(values, outputShape, detectionProfile);
        float[][] keypointRows = ReadPerBoxAuxiliaryRows(
            values,
            outputShape,
            detectionProfile.Postprocess,
            channelStart,
            auxiliaryWidth,
            metadata.AuxiliaryLayout);

        List<YoloPosePrediction> poses = new List<YoloPosePrediction>();
        foreach (YoloDetection detection in detections)
        {
            if (detection.SourceIndex < 0 || detection.SourceIndex >= keypointRows.Length)
            {
                continue;
            }

            YoloPoseKeypoint[] keypoints = YoloPoseDecoder.DecodeFlatKeypoints(
                keypointRows[detection.SourceIndex],
                metadata.PoseKeypointCount,
                metadata.PoseKeypointStride);
            poses.Add(new YoloPosePrediction(detection, keypoints));
        }

        return YoloVisionResult.FromPoses(poses, "Decoded pose keypoints from channels embedded in the detection tensor.");
    }

    public static YoloVisionResult DecodeObbOutputs(
        float[] boxValues,
        int[] boxShape,
        float[] angleValues,
        int[] angleShape,
        YoloModelProfile profile,
        YoloMultiOutputMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        YoloModelProfile detectionProfile = CreateObbCandidateProfile(CreateDetectionProfile(profile));
        IReadOnlyList<YoloDetection> detections = DecodeDetections(boxValues, boxShape, detectionProfile);
        float[][] angleRows = ReadAuxiliaryRows(angleValues, angleShape, 1, metadata.AuxiliaryLayout);

        List<YoloObbDetection> orientedBoxes = new List<YoloObbDetection>();
        foreach (YoloDetection detection in detections)
        {
            if (detection.SourceIndex < 0 || detection.SourceIndex >= angleRows.Length)
            {
                continue;
            }

            orientedBoxes.Add(YoloObbDecoder.Decode(detection, angleRows[detection.SourceIndex][0], metadata.ObbAngleInDegrees));
        }

        return YoloVisionResult.FromOrientedBoxes(
            ApplyObbPostprocess(orientedBoxes, profile.Postprocess),
            "Decoded oriented boxes from detection rows and angle tensor using probabilistic-IoU rotated NMS.");
    }

    public static YoloVisionResult DecodeEmbeddedObbOutput(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile,
        YoloMultiOutputMetadata metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        const int angleWidth = 1;
        YoloModelProfile detectionProfile = CreateObbCandidateProfile(
            CreateDetectionProfileForAuxiliary(profile, outputShape, angleWidth));
        int channelStart = ValidateEmbeddedAuxiliaryContract(
            outputShape,
            detectionProfile.Postprocess,
            metadata,
            angleWidth,
            "OBB angle");
        IReadOnlyList<YoloDetection> detections = DecodeDetections(values, outputShape, detectionProfile);
        float[][] angleRows = ReadPerBoxAuxiliaryRows(
            values,
            outputShape,
            detectionProfile.Postprocess,
            channelStart,
            angleWidth,
            metadata.AuxiliaryLayout);

        List<YoloObbDetection> orientedBoxes = new List<YoloObbDetection>();
        foreach (YoloDetection detection in detections)
        {
            if (detection.SourceIndex < 0 || detection.SourceIndex >= angleRows.Length)
            {
                continue;
            }

            orientedBoxes.Add(YoloObbDecoder.Decode(
                detection,
                angleRows[detection.SourceIndex][0],
                metadata.ObbAngleInDegrees));
        }

        return YoloVisionResult.FromOrientedBoxes(
            ApplyObbPostprocess(orientedBoxes, profile.Postprocess),
            "Decoded oriented boxes from angle channels embedded in the detection tensor using probabilistic-IoU rotated NMS.");
    }

    private static YoloVisionResult DecodeSingleRoleOutput(YoloRuntimeOutputSet outputs, YoloModelProfile profile, YoloOutputTensorRole role)
    {
        YoloRuntimeOutputTensor output = outputs.GetRequired(role);
        return DecodeOutput(output.Values, output.Shape, profile);
    }

    private static YoloVisionResult DecodeSegmentationRuntimeOutputs(YoloRuntimeOutputSet outputs, YoloModelProfile profile, YoloMultiOutputMetadata? metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata), "Segmentation runtime outputs require YoloMultiOutputMetadata.");
        }

        YoloRuntimeOutputTensor detection = outputs.GetRequired(YoloOutputTensorRole.Detection);
        YoloRuntimeOutputTensor prototypes = outputs.GetRequired(YoloOutputTensorRole.MaskPrototypes);
        return DecodeSegmentationOutputs(detection.Values, detection.Shape, prototypes.Values, prototypes.Shape, profile, metadata);
    }

    private static YoloVisionResult DecodePoseRuntimeOutputs(YoloRuntimeOutputSet outputs, YoloModelProfile profile, YoloMultiOutputMetadata? metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata), "Pose runtime outputs require YoloMultiOutputMetadata.");
        }

        YoloRuntimeOutputTensor detection = outputs.GetRequired(YoloOutputTensorRole.Detection);
        YoloRuntimeOutputTensor? keypoints = outputs.TryGet(YoloOutputTensorRole.PoseKeypoints);
        return keypoints == null
            ? DecodeEmbeddedPoseOutput(detection.Values, detection.Shape, profile, metadata)
            : DecodePoseOutputs(detection.Values, detection.Shape, keypoints.Values, keypoints.Shape, profile, metadata);
    }

    private static YoloVisionResult DecodeObbRuntimeOutputs(YoloRuntimeOutputSet outputs, YoloModelProfile profile, YoloMultiOutputMetadata? metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata), "OBB runtime outputs require YoloMultiOutputMetadata.");
        }

        YoloRuntimeOutputTensor detection = outputs.GetRequired(YoloOutputTensorRole.Detection);
        YoloRuntimeOutputTensor? angles = outputs.TryGet(YoloOutputTensorRole.ObbAngles);
        return angles == null
            ? DecodeEmbeddedObbOutput(detection.Values, detection.Shape, profile, metadata)
            : DecodeObbOutputs(detection.Values, detection.Shape, angles.Values, angles.Shape, profile, metadata);
    }

    public static IReadOnlyList<YoloClassificationPrediction> DecodeClassifications(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        int classCount = GetClassificationClassCount(values, outputShape, profile);
        return values
            .Take(classCount)
            .Select(static (score, index) => new YoloClassificationPrediction(index, score))
            .Where(item => item.Score >= profile.Postprocess.ConfidenceThreshold)
            .OrderByDescending(static item => item.Score)
            .Take(profile.Postprocess.TopK)
            .ToArray();
    }

    public static YoloSemanticMap DecodeSemanticMap(
        float[] values,
        int[] outputShape,
        YoloModelProfile profile)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (outputShape == null)
        {
            throw new ArgumentNullException(nameof(outputShape));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        ValidateValueCount(values, outputShape);
        int configuredClassCount = profile.Postprocess.ClassCount;
        if (outputShape.Length == 3)
        {
            int classes = outputShape[0];
            int height = outputShape[1];
            int width = outputShape[2];
            return new YoloSemanticMap(classes, width, height, (float[])values.Clone());
        }

        if (outputShape.Length != 4 || outputShape[0] != 1)
        {
            throw new NotSupportedException("Semantic segmentation output is expected to be [1,C,H,W], [1,H,W,C], or [C,H,W].");
        }

        if (configuredClassCount > 0 && outputShape[3] == configuredClassCount && outputShape[1] != configuredClassCount)
        {
            return DecodeNhwcSemanticMap(values, outputShape);
        }

        int classCount = outputShape[1];
        int heightNchw = outputShape[2];
        int widthNchw = outputShape[3];
        return new YoloSemanticMap(classCount, widthNchw, heightNchw, (float[])values.Clone());
    }

    private static int GetClassificationClassCount(float[] values, int[] outputShape, YoloModelProfile profile)
    {
        if (outputShape == null)
        {
            throw new ArgumentNullException(nameof(outputShape));
        }

        ValidateValueCount(values, outputShape);
        int configuredClassCount = profile.Postprocess.ClassCount;
        if (outputShape.Length == 1)
        {
            return configuredClassCount > 0 ? Math.Min(configuredClassCount, outputShape[0]) : outputShape[0];
        }

        if (outputShape.Length == 2 && outputShape[0] == 1)
        {
            return configuredClassCount > 0 ? Math.Min(configuredClassCount, outputShape[1]) : outputShape[1];
        }

        if (outputShape.Length == 2 && outputShape[1] == 1)
        {
            return configuredClassCount > 0 ? Math.Min(configuredClassCount, outputShape[0]) : outputShape[0];
        }

        throw new NotSupportedException("Classification output is expected to be [C], [1,C], or [C,1].");
    }

    private static YoloModelProfile CreateDetectionProfile(YoloModelProfile profile)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        return new YoloModelProfile(
            profile.Family,
            YoloTaskType.Detection,
            profile.InputName,
            profile.OutputName,
            profile.InputShape,
            profile.Preprocess,
            profile.Postprocess);
    }

    private static YoloModelProfile CreateDetectionProfileForAuxiliary(YoloModelProfile profile, int[] outputShape, int auxiliaryWidth)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        bool? hasObjectness = profile.Postprocess.HasObjectness;
        if (!hasObjectness.HasValue)
        {
            hasObjectness = InferHasObjectnessForAuxiliary(outputShape, profile.Postprocess, auxiliaryWidth);
        }

        YoloPostprocessOptions postprocess = new YoloPostprocessOptions(
            profile.Postprocess.Layout,
            hasObjectness,
            profile.Postprocess.ClassCount,
            profile.Postprocess.ConfidenceThreshold,
            profile.Postprocess.IouThreshold,
            profile.Postprocess.TopK,
            profile.Postprocess.ApplyNms,
            profile.Postprocess.NmsMode);

        return new YoloModelProfile(
            profile.Family,
            YoloTaskType.Detection,
            profile.InputName,
            profile.OutputName,
            profile.InputShape,
            profile.Preprocess,
            postprocess);
    }

    private static YoloModelProfile CreateObbCandidateProfile(YoloModelProfile profile)
    {
        YoloPostprocessOptions source = profile.Postprocess;
        YoloPostprocessOptions candidatePostprocess = new YoloPostprocessOptions(
            source.Layout,
            source.HasObjectness,
            source.ClassCount,
            source.ConfidenceThreshold,
            source.IouThreshold,
            int.MaxValue,
            applyNms: false,
            nmsMode: YoloNmsMode.None);
        return new YoloModelProfile(
            profile.Family,
            YoloTaskType.Detection,
            profile.InputName,
            profile.OutputName,
            profile.InputShape,
            profile.Preprocess,
            candidatePostprocess);
    }

    private static IReadOnlyList<YoloObbDetection> ApplyObbPostprocess(
        IEnumerable<YoloObbDetection> detections,
        YoloPostprocessOptions postprocess)
    {
        IReadOnlyList<YoloObbDetection> ranked = detections
            .OrderByDescending(static item => item.Box.Score)
            .ToArray();
        if (postprocess.ApplyNms && postprocess.NmsMode != YoloNmsMode.None)
        {
            ranked = YoloObbDecoder.ApplyFastNms(
                ranked,
                postprocess.IouThreshold,
                classAware: postprocess.NmsMode != YoloNmsMode.ClassAgnostic);
        }

        return ranked
            .OrderByDescending(static item => item.Box.Score)
            .Take(postprocess.TopK)
            .ToArray();
    }

    private static bool InferHasObjectnessForAuxiliary(int[] outputShape, YoloPostprocessOptions postprocess, int auxiliaryWidth)
    {
        if (postprocess.ClassCount <= 0)
        {
            throw new NotSupportedException("Class count must be known before objectness can be inferred for an auxiliary-channel output.");
        }

        if (auxiliaryWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(auxiliaryWidth), "Auxiliary width must be positive.");
        }

        YoloOutputLayout resolvedLayout = YoloOutputLayoutInference.InferRank3(outputShape, postprocess.Layout);
        int channelCount = resolvedLayout == YoloOutputLayout.ChannelsFirst ? outputShape[1] : outputShape[2];
        int noObjectnessChannels = BoxChannelCount + postprocess.ClassCount + auxiliaryWidth;
        int objectnessChannels = BoxChannelCount + 1 + postprocess.ClassCount + auxiliaryWidth;

        if (channelCount == noObjectnessChannels)
        {
            return false;
        }

        if (channelCount == objectnessChannels)
        {
            return true;
        }

        throw new NotSupportedException("Auxiliary-channel output must either set --has-objectness explicitly or match box + class + auxiliary channel counts exactly.");
    }

    private static float[][] ReadPerBoxAuxiliaryRows(
        float[] values,
        int[] outputShape,
        YoloPostprocessOptions postprocess,
        int? auxiliaryChannelStart,
        int auxiliaryWidth,
        YoloOutputLayout requestedLayout)
    {
        if (postprocess == null)
        {
            throw new ArgumentNullException(nameof(postprocess));
        }

        int channelStart = auxiliaryChannelStart ?? ResolveAuxiliaryChannelStart(values, outputShape, postprocess, auxiliaryWidth);
        YoloOutputLayout effectiveLayout = requestedLayout == YoloOutputLayout.Auto ? postprocess.Layout : requestedLayout;
        return ReadAuxiliaryRows(values, outputShape, auxiliaryWidth, effectiveLayout, channelStart);
    }

    private static int ResolveAuxiliaryChannelStart(float[] values, int[] outputShape, YoloPostprocessOptions postprocess, int auxiliaryWidth)
    {
        YoloOutputLayout resolvedLayout = YoloOutputLayoutInference.InferRank3(outputShape, postprocess.Layout);
        int channelCount = resolvedLayout == YoloOutputLayout.ChannelsFirst ? outputShape[1] : outputShape[2];
        bool hasObjectness = postprocess.HasObjectness ?? InferHasObjectnessForAuxiliary(outputShape, postprocess, auxiliaryWidth);
        int classCount = postprocess.ClassCount > 0 ? postprocess.ClassCount : channelCount - (hasObjectness ? 5 : 4);
        if (classCount <= 0)
        {
            throw new NotSupportedException("Class count must be known before auxiliary channels can be sliced from a detection output.");
        }

        _ = values;
        return BoxChannelCount + (hasObjectness ? 1 : 0) + classCount;
    }

    private static int ValidateEmbeddedAuxiliaryContract(
        int[] outputShape,
        YoloPostprocessOptions postprocess,
        YoloMultiOutputMetadata metadata,
        int auxiliaryWidth,
        string auxiliaryName = "pose keypoint")
    {
        if (postprocess.ClassCount <= 0)
        {
            throw new NotSupportedException($"Embedded {auxiliaryName} output requires a known class count so detection and auxiliary channels cannot be confused.");
        }

        YoloOutputLayout detectionLayout = YoloOutputLayoutInference.InferRank3(outputShape, postprocess.Layout);
        YoloOutputLayout requestedAuxiliaryLayout = metadata.AuxiliaryLayout == YoloOutputLayout.Auto
            ? postprocess.Layout
            : metadata.AuxiliaryLayout;
        YoloOutputLayout auxiliaryLayout = YoloOutputLayoutInference.InferRank3(outputShape, requestedAuxiliaryLayout);
        if (auxiliaryLayout != detectionLayout)
        {
            throw new NotSupportedException($"Embedded {auxiliaryName} channels must use the same tensor layout as the detection channels.");
        }

        int channelCount = detectionLayout == YoloOutputLayout.ChannelsFirst ? outputShape[1] : outputShape[2];
        bool hasObjectness = postprocess.HasObjectness ?? InferHasObjectnessForAuxiliary(outputShape, postprocess, auxiliaryWidth);
        int expectedChannelStart = BoxChannelCount + (hasObjectness ? 1 : 0) + postprocess.ClassCount;
        int channelStart = metadata.AuxiliaryChannelStart ?? expectedChannelStart;
        if (channelStart != expectedChannelStart)
        {
            throw new NotSupportedException($"Embedded {auxiliaryName} channel start {channelStart} does not match the detection prefix width {expectedChannelStart}.");
        }

        if (channelStart + auxiliaryWidth != channelCount)
        {
            throw new NotSupportedException($"Embedded {auxiliaryName} output has {channelCount} channels, but detection prefix {channelStart} and auxiliary width {auxiliaryWidth} require exactly {channelStart + auxiliaryWidth}.");
        }

        return channelStart;
    }

    private static float[][] ReadAuxiliaryRows(float[] values, int[] outputShape, int auxiliaryWidth, YoloOutputLayout requestedLayout, int channelStart = 0)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (outputShape == null)
        {
            throw new ArgumentNullException(nameof(outputShape));
        }

        if (auxiliaryWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(auxiliaryWidth), "Auxiliary width must be positive.");
        }

        YoloOutputLayout resolvedLayout = YoloOutputLayoutInference.InferRank3(outputShape, requestedLayout);
        bool channelsFirst = resolvedLayout == YoloOutputLayout.ChannelsFirst;
        int channelCount = channelsFirst ? outputShape[1] : outputShape[2];
        int boxCount = channelsFirst ? outputShape[2] : outputShape[1];
        if (values.Length != checked(channelCount * boxCount))
        {
            throw new ArgumentException("Output value count does not match the output tensor shape.", nameof(values));
        }

        if (channelStart < 0 || channelStart + auxiliaryWidth > channelCount)
        {
            throw new ArgumentException("Auxiliary channel range is outside the output tensor channel count.", nameof(channelStart));
        }

        float[][] rows = new float[boxCount][];
        for (int box = 0; box < boxCount; box++)
        {
            float[] row = new float[auxiliaryWidth];
            for (int channel = 0; channel < auxiliaryWidth; channel++)
            {
                int absoluteChannel = channelStart + channel;
                int sourceIndex = channelsFirst
                    ? absoluteChannel * boxCount + box
                    : box * channelCount + absoluteChannel;
                row[channel] = values[sourceIndex];
            }

            rows[box] = row;
        }

        return rows;
    }

    private sealed class PrototypeTensor
    {
        private PrototypeTensor(float[] values, int prototypeCount, int width, int height)
        {
            Values = values;
            PrototypeCount = prototypeCount;
            Width = width;
            Height = height;
        }

        public float[] Values { get; }

        public int PrototypeCount { get; }

        public int Width { get; }

        public int Height { get; }

        public static PrototypeTensor FromValues(float[] values, int[] shape)
        {
            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

            if (shape == null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            ValidateValueCount(values, shape);
            if (shape.Length == 3)
            {
                return new PrototypeTensor((float[])values.Clone(), shape[0], shape[2], shape[1]);
            }

            if (shape.Length == 4 && shape[0] == 1)
            {
                return new PrototypeTensor((float[])values.Clone(), shape[1], shape[3], shape[2]);
            }

            throw new NotSupportedException("Mask prototype tensor is expected to be [P,H,W] or [1,P,H,W].");
        }
    }

    private static YoloSemanticMap DecodeNhwcSemanticMap(float[] values, int[] outputShape)
    {
        int height = outputShape[1];
        int width = outputShape[2];
        int classCount = outputShape[3];
        float[] classMajor = new float[values.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < classCount; c++)
                {
                    int source = ((y * width) + x) * classCount + c;
                    int target = ((c * height) + y) * width + x;
                    classMajor[target] = values[source];
                }
            }
        }

        return new YoloSemanticMap(classCount, width, height, classMajor);
    }

    private static void ValidateValueCount(float[] values, int[] outputShape)
    {
        int expected = 1;
        for (int index = 0; index < outputShape.Length; index++)
        {
            if (outputShape[index] <= 0)
            {
                throw new ArgumentException("Output shape dimensions must be positive.", nameof(outputShape));
            }

            expected = checked(expected * outputShape[index]);
        }

        if (values.Length != expected)
        {
            throw new ArgumentException("Output value count does not match the output tensor shape.", nameof(values));
        }
    }
}
