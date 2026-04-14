namespace WinFormsAppDemo;

internal enum YoloTaskType
{
    Detect,
    Segment,
    Pose,
    Obb
}

internal sealed class YoloModelProfile
{
    private YoloModelProfile(
        string displayName,
        YoloTaskType taskType,
        bool hasObjectness,
        bool outputsAlreadyNms,
        int poseKeypointCount,
        float confidenceThreshold,
        float nmsThreshold,
        float maskThreshold,
        float keypointThreshold)
    {
        DisplayName = displayName;
        TaskType = taskType;
        HasObjectness = hasObjectness;
        OutputsAlreadyNms = outputsAlreadyNms;
        PoseKeypointCount = poseKeypointCount;
        ConfidenceThreshold = confidenceThreshold;
        NmsThreshold = nmsThreshold;
        MaskThreshold = maskThreshold;
        KeypointThreshold = keypointThreshold;
    }

    public string DisplayName { get; }

    public YoloTaskType TaskType { get; }

    public bool HasObjectness { get; }

    public bool OutputsAlreadyNms { get; }

    public float ConfidenceThreshold { get; }

    public float NmsThreshold { get; }

    public float MaskThreshold { get; }

    public float KeypointThreshold { get; }

    public int PoseKeypointCount { get; }

    public static IReadOnlyList<YoloModelProfile> All { get; } =
    [
        new("YOLOv5 Detect", YoloTaskType.Detect, true, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv5 Segment", YoloTaskType.Segment, true, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv5u Detect", YoloTaskType.Detect, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv5u Segment", YoloTaskType.Segment, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv8 Detect", YoloTaskType.Detect, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv8 Segment", YoloTaskType.Segment, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv8 Pose", YoloTaskType.Pose, false, false, 17, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv8 OBB", YoloTaskType.Obb, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv11 Detect", YoloTaskType.Detect, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv11 Segment", YoloTaskType.Segment, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv11 Pose", YoloTaskType.Pose, false, false, 17, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLOv11 OBB", YoloTaskType.Obb, false, false, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLO26 Detect", YoloTaskType.Detect, false, true, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLO26 Segment", YoloTaskType.Segment, false, true, 0, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLO26 Pose", YoloTaskType.Pose, false, true, 17, 0.25f, 0.45f, 0.50f, 0.50f),
        new("YOLO26 OBB", YoloTaskType.Obb, false, true, 0, 0.25f, 0.45f, 0.50f, 0.50f)
    ];

    public static YoloModelProfile Default => All.First(x => x.DisplayName == "YOLOv11 Detect");

    public static YoloModelProfile? TryMatchByPath(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            return null;
        }

        string fileName = Path.GetFileNameWithoutExtension(modelPath)
            .ToLowerInvariant()
            .Replace("-", string.Empty)
            .Replace("_", string.Empty);

        YoloTaskType taskType = ResolveTaskType(fileName);

        if (fileName.Contains("yolo26", StringComparison.Ordinal))
        {
            return Find("YOLO26", taskType);
        }

        if (fileName.Contains("yolov5u", StringComparison.Ordinal) || fileName.Contains("yolo5u", StringComparison.Ordinal))
        {
            return Find("YOLOv5u", taskType);
        }

        if (fileName.Contains("yolov5", StringComparison.Ordinal) || fileName.Contains("yolo5", StringComparison.Ordinal))
        {
            return Find("YOLOv5", taskType);
        }

        if (fileName.Contains("yolov8", StringComparison.Ordinal) || fileName.Contains("yolo8", StringComparison.Ordinal))
        {
            return Find("YOLOv8", taskType);
        }

        if (fileName.Contains("yolov11", StringComparison.Ordinal) || fileName.Contains("yolo11", StringComparison.Ordinal))
        {
            return Find("YOLOv11", taskType);
        }

        return null;
    }

    private static YoloTaskType ResolveTaskType(string fileName)
    {
        if (fileName.Contains("seg", StringComparison.Ordinal))
        {
            return YoloTaskType.Segment;
        }

        if (fileName.Contains("pose", StringComparison.Ordinal))
        {
            return YoloTaskType.Pose;
        }

        if (fileName.Contains("obb", StringComparison.Ordinal))
        {
            return YoloTaskType.Obb;
        }

        return YoloTaskType.Detect;
    }

    private static YoloModelProfile? Find(string versionPrefix, YoloTaskType taskType)
    {
        return All.FirstOrDefault(x => x.DisplayName.StartsWith(versionPrefix, StringComparison.Ordinal) && x.TaskType == taskType);
    }

    public override string ToString()
    {
        return DisplayName;
    }
}
