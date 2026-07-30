using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp;

namespace YoloVisionSample;

/// <summary>
/// Context values written into a YoloVision output JSON report.
/// 写入 YoloVision 输出 JSON 报告的上下文字段。
/// </summary>
public sealed class YoloVisionOutputReportContext
{
    /// <summary>
    /// Creates output-report context values.
    /// 创建输出报告上下文字段。
    /// </summary>
    public YoloVisionOutputReportContext(
        string modelPath,
        string inputPath,
        string inputDataPath,
        string inputPattern,
        int[] inputShape,
        int tensorRtLine,
        int profileIndex,
        ulong engineDeviceMemoryBytes,
        double elapsedMilliseconds,
        string labelsPath = "",
        YoloImagePreprocessResult? imagePreprocess = null)
        : this(
            modelPath,
            inputPath,
            inputDataPath,
            inputPattern,
            inputShape,
            tensorRtLine,
            profileIndex,
            engineDeviceMemoryBytes,
            elapsedMilliseconds,
            labelsPath,
            imagePreprocess,
            segmentationSpatialTransform: null)
    {
    }

    public YoloVisionOutputReportContext(
        string modelPath,
        string inputPath,
        string inputDataPath,
        string inputPattern,
        int[] inputShape,
        int tensorRtLine,
        int profileIndex,
        ulong engineDeviceMemoryBytes,
        double elapsedMilliseconds,
        string labelsPath,
        YoloImagePreprocessResult? imagePreprocess,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform)
    {
        ModelPath = modelPath ?? string.Empty;
        InputPath = inputPath ?? string.Empty;
        InputDataPath = inputDataPath ?? string.Empty;
        InputPattern = string.IsNullOrWhiteSpace(inputPattern) ? "ramp" : inputPattern;
        InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
        TensorRtLine = tensorRtLine;
        ProfileIndex = profileIndex;
        EngineDeviceMemoryBytes = engineDeviceMemoryBytes;
        ElapsedMilliseconds = elapsedMilliseconds;
        LabelsPath = labelsPath ?? string.Empty;
        ImagePreprocess = imagePreprocess;
        if (segmentationSpatialTransform != null && imagePreprocess == null)
        {
            throw new ArgumentException(
                "Segmentation spatial transform requires image preprocessing metadata.",
                nameof(segmentationSpatialTransform));
        }

        SegmentationSpatialTransform = segmentationSpatialTransform;
    }

    /// <summary>Gets the ONNX model path. 获取 ONNX 模型路径。</summary>
    public string ModelPath { get; }

    /// <summary>Gets the raw byte tensor input path, when supplied. 获取 raw byte tensor 输入路径。</summary>
    public string InputPath { get; }

    /// <summary>Gets the float tensor input path, when supplied. 获取 float tensor 输入路径。</summary>
    public string InputDataPath { get; }

    /// <summary>Gets the synthetic input pattern when no external tensor is supplied. 获取未提供外部 tensor 时的合成输入模式。</summary>
    public string InputPattern { get; }

    /// <summary>Gets the concrete input tensor shape. 获取具体输入 tensor shape。</summary>
    public int[] InputShape { get; }

    /// <summary>Gets the TensorRT API line used by the run. 获取本次运行使用的 TensorRT API line。</summary>
    public int TensorRtLine { get; }

    /// <summary>Gets the TensorRT optimization profile index. 获取 TensorRT optimization profile index。</summary>
    public int ProfileIndex { get; }

    /// <summary>Gets engine device-memory size reported by TensorRT. 获取 TensorRT 报告的 engine device memory 大小。</summary>
    public ulong EngineDeviceMemoryBytes { get; }

    /// <summary>Gets measured enqueue elapsed milliseconds. 获取测量得到的 enqueue 耗时。</summary>
    public double ElapsedMilliseconds { get; }

    /// <summary>Gets the labels file path, when supplied. 获取 labels 文件路径。</summary>
    public string LabelsPath { get; }

    /// <summary>Gets image preprocessing metadata, when this run used <c>--image</c>. 获取 --image 预处理元数据。</summary>
    public YoloImagePreprocessResult? ImagePreprocess { get; }

    /// <summary>Gets the explicit segmentation spatial-transform options, when requested. 获取显式 segmentation 空间变换选项。</summary>
    public YoloSegmentationSpatialTransformOptions? SegmentationSpatialTransform { get; }

    internal IReadOnlyList<OnnxSampleInputTensor> RuntimeInputTensors { get; private set; } = Array.Empty<OnnxSampleInputTensor>();

    internal OnnxSampleReferenceValidationResult? RuntimeReferenceValidation { get; private set; }

    internal void AttachRuntimeEvidence(OnnxSampleMultiOutputResult run)
    {
        RuntimeInputTensors = run?.Inputs ?? throw new ArgumentNullException(nameof(run));
        RuntimeReferenceValidation = run.ReferenceValidation;
    }
}

/// <summary>
/// Writes pointer-free YoloVision output JSON reports for owner evidence and golden-output review.
/// 为 owner evidence 和 golden-output 复查写出无指针的 YoloVision 输出 JSON 报告。
/// </summary>
public static class YoloVisionOutputReport
{
    public const string SchemaVersion = "yolovision-output.v1";
    private const string BoundaryEvidenceKind = "YoloVision output schema; readonly diagnostics; not runtime proof";

    private static readonly JsonWriterOptions WriterOptions = new JsonWriterOptions { Indented = true };

    /// <summary>
    /// Serializes a YoloVision output report to JSON.
    /// 将 YoloVision 输出报告序列化为 JSON。
    /// </summary>
    public static string ToJson(
        YoloVisionOutputReportContext context,
        YoloRuntimeOutputSet outputs,
        YoloModelProfile profile,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        TensorRtEngineBindingReport? bindingReport = null)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (context.SegmentationSpatialTransform != null && result.TaskType != YoloTaskType.Segmentation)
        {
            throw new ArgumentException(
                "Segmentation spatial transform can only be written for a segmentation result.",
                nameof(result));
        }

        using MemoryStream stream = new MemoryStream();
        using (Utf8JsonWriter writer = new Utf8JsonWriter(stream, WriterOptions))
        {
            WriteReport(writer, context, outputs, profile, result, labels ?? Array.Empty<string>(), bindingReport);
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    /// <summary>
    /// Writes a YoloVision output report to disk.
    /// 将 YoloVision 输出报告写入磁盘。
    /// </summary>
    public static void Write(
        string outputPath,
        YoloVisionOutputReportContext context,
        YoloRuntimeOutputSet outputs,
        YoloModelProfile profile,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        TensorRtEngineBindingReport? bindingReport = null)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output JSON path must not be empty.", nameof(outputPath));
        }

        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(fullPath, ToJson(context, outputs, profile, result, labels, bindingReport), Encoding.UTF8);
    }

    internal static void Write(
        string outputPath,
        OnnxSampleOptions options,
        OnnxSampleMultiOutputResult run,
        YoloRuntimeOutputSet outputs,
        YoloModelProfile profile,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        string labelsPath = "",
        YoloImagePreprocessResult? imagePreprocess = null,
        YoloSegmentationSpatialTransformOptions? segmentationSpatialTransform = null)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (run == null)
        {
            throw new ArgumentNullException(nameof(run));
        }

        YoloVisionOutputReportContext context = new YoloVisionOutputReportContext(
                options.ModelPath,
                options.InputPath,
                options.InputDataPath,
                options.InputPattern,
                options.InputShape.Values,
                (int)run.Line,
                run.ProfileIndex,
                run.EngineDeviceMemoryBytes,
                run.ElapsedMilliseconds,
                labelsPath,
                imagePreprocess,
                segmentationSpatialTransform);
        context.AttachRuntimeEvidence(run);
        Write(
            outputPath,
            context,
            outputs,
            profile,
            result,
            labels,
            run.Report);
    }

    private static void WriteReport(
        Utf8JsonWriter writer,
        YoloVisionOutputReportContext context,
        YoloRuntimeOutputSet outputs,
        YoloModelProfile profile,
        YoloVisionResult result,
        IReadOnlyList<string> labels,
        TensorRtEngineBindingReport? bindingReport)
    {
        writer.WriteStartObject();
        writer.WriteString("schemaVersion", SchemaVersion);
        writer.WriteString("task", ToTaskAlias(result.TaskType));
        writer.WriteString("modelFamily", ToFamilyAlias(profile.Family));
        WriteLabels(writer, context, labels);
        WriteInput(writer, context);
        WriteInputTensors(writer, context.RuntimeInputTensors);
        WriteEngine(writer, context);
        WriteRuntime(writer, context);
        WriteOutputs(writer, outputs);
        WriteReferenceValidation(writer, context.RuntimeReferenceValidation);
        if (bindingReport != null)
        {
            WriteBindingMetadata(writer, bindingReport, outputs);
        }
        WritePostprocess(writer, profile);
        WritePredictions(writer, context, result, labels);
        WriteBoundary(writer);
        writer.WriteEndObject();
    }

    private static void WriteLabels(Utf8JsonWriter writer, YoloVisionOutputReportContext context, IReadOnlyList<string> labels)
    {
        writer.WritePropertyName("labels");
        writer.WriteStartObject();
        writer.WriteString("path", context.LabelsPath);
        writer.WriteString("sha256", ComputeFileSha256OrEmpty(context.LabelsPath));
        writer.WriteNumber("classCount", labels?.Count ?? 0);
        writer.WriteString("license", "owner-record-required");
        writer.WriteEndObject();
    }

    private static void WriteInput(Utf8JsonWriter writer, YoloVisionOutputReportContext context)
    {
        string inputSourcePath = context.ImagePreprocess != null
            ? context.ImagePreprocess.TensorPath
            : !string.IsNullOrWhiteSpace(context.InputDataPath)
            ? context.InputDataPath
            : context.InputPath;
        int width = context.ImagePreprocess?.TargetWidth ?? (context.InputShape.Length >= 4 ? context.InputShape[3] : 1);
        int height = context.ImagePreprocess?.TargetHeight ?? (context.InputShape.Length >= 4 ? context.InputShape[2] : 1);

        writer.WritePropertyName("input");
        writer.WriteStartObject();
        writer.WriteString("path", string.IsNullOrWhiteSpace(inputSourcePath) ? context.InputPattern : inputSourcePath);
        writer.WriteString("sha256", ComputeFileSha256OrEmpty(inputSourcePath));
        writer.WriteNumber("width", width);
        writer.WriteNumber("height", height);
        writer.WriteString("sourceKind", context.ImagePreprocess != null ? "preprocessed-image-tensor" : string.IsNullOrWhiteSpace(inputSourcePath) ? "synthetic-pattern" : "external-tensor");
        if (context.ImagePreprocess != null)
        {
            WriteImagePreprocess(writer, context.ImagePreprocess);
        }

        writer.WriteEndObject();
    }

    private static void WriteInputTensors(Utf8JsonWriter writer, IReadOnlyList<OnnxSampleInputTensor> inputs)
    {
        writer.WritePropertyName("inputTensors");
        writer.WriteStartArray();
        foreach (OnnxSampleInputTensor input in inputs)
        {
            writer.WriteStartObject();
            writer.WriteString("tensorName", input.Name);
            writer.WritePropertyName("shape");
            WriteIntArray(writer, input.Shape.Values);
            writer.WriteNumber("elementCount", input.ElementCount);
            writer.WriteNumber("byteLength", input.ByteLength);
            writer.WriteString("sha256", input.Sha256);
            writer.WriteString("sourceClassification", input.SourceClassification);
            writer.WriteString("sourcePath", input.SourcePath);
            writer.WritePropertyName("preview");
            WriteFloatPreview(writer, input.Preview.ToArray());
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteReferenceValidation(Utf8JsonWriter writer, OnnxSampleReferenceValidationResult? validation)
    {
        writer.WritePropertyName("referenceValidation");
        writer.WriteStartObject();
        writer.WriteBoolean("requested", validation?.Requested ?? false);
        writer.WriteBoolean("completed", validation?.Completed ?? false);
        writer.WriteBoolean("passed", validation?.Passed ?? false);
        writer.WriteNumber("absoluteTolerance", validation?.AbsoluteTolerance ?? 0.0f);
        writer.WriteNumber("relativeTolerance", validation?.RelativeTolerance ?? 0.0f);
        writer.WriteString("nanPolicy", validation?.NaNPolicy ?? "reject");
        writer.WriteString("infinityPolicy", validation?.InfinityPolicy ?? "exact");
        writer.WritePropertyName("tensorComparisons");
        writer.WriteStartArray();
        foreach (OnnxSampleReferenceTensorComparison comparison in validation?.TensorComparisons ?? Array.Empty<OnnxSampleReferenceTensorComparison>())
        {
            writer.WriteStartObject();
            writer.WriteString("tensorName", comparison.TensorName);
            writer.WriteString("referencePath", comparison.ReferencePath);
            writer.WriteString("referenceSha256", comparison.ReferenceSha256);
            writer.WriteString("sourceClassification", comparison.SourceClassification);
            writer.WritePropertyName("actualShape");
            WriteIntArray(writer, comparison.ActualShape.ToArray());
            writer.WritePropertyName("referenceShape");
            WriteIntArray(writer, comparison.ReferenceShape.ToArray());
            writer.WriteNumber("actualElementCount", comparison.ActualElementCount);
            writer.WriteNumber("referenceElementCount", comparison.ReferenceElementCount);
            writer.WriteNumber("comparedElementCount", comparison.ComparedElementCount);
            writer.WriteNumber("mismatchCount", comparison.MismatchCount);
            writer.WriteNumber("firstMismatchIndex", comparison.FirstMismatchIndex);
            WriteFloatProperty(writer, "maximumAbsoluteError", comparison.MaximumAbsoluteError);
            WriteFloatProperty(writer, "maximumRelativeError", comparison.MaximumRelativeError);
            writer.WriteBoolean("completed", comparison.Completed);
            writer.WriteBoolean("passed", comparison.Passed);
            writer.WriteString("diagnostic", comparison.Diagnostic);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WritePropertyName("diagnostics");
        writer.WriteStartArray();
        foreach (string diagnostic in validation?.Diagnostics ?? Array.Empty<string>())
        {
            writer.WriteStringValue(diagnostic);
        }
        writer.WriteEndArray();
        writer.WriteString("proofBoundary", "Structured reference comparison proves only the recorded tensor values under the declared policy; synthetic references and hashes do not promote real-model or package-consumer proof.");
        writer.WriteEndObject();
    }

    private static void WriteImagePreprocess(Utf8JsonWriter writer, YoloImagePreprocessResult imagePreprocess)
    {
        writer.WritePropertyName("image");
        writer.WriteStartObject();
        writer.WriteString("path", imagePreprocess.SourcePath);
        writer.WriteString("sha256", imagePreprocess.SourceSha256);
        writer.WriteNumber("width", imagePreprocess.SourceWidth);
        writer.WriteNumber("height", imagePreprocess.SourceHeight);
        writer.WriteEndObject();

        writer.WritePropertyName("preprocessedTensor");
        writer.WriteStartObject();
        writer.WriteString("path", imagePreprocess.TensorPath);
        writer.WriteString("sha256", imagePreprocess.TensorSha256);
        writer.WriteNumber("elementCount", imagePreprocess.TensorElementCount);
        writer.WriteString("layout", imagePreprocess.TensorLayout);
        writer.WriteString("colorOrder", imagePreprocess.ColorOrder);
        writer.WriteBoolean("normalized", imagePreprocess.Normalized);
        writer.WriteNumber("scale", imagePreprocess.Scale);
        writer.WriteEndObject();

        writer.WritePropertyName("letterbox");
        writer.WriteStartObject();
        writer.WriteBoolean("enabled", imagePreprocess.LetterboxEnabled);
        writer.WriteString("resizeMode", imagePreprocess.ResizeMode);
        writer.WriteString("alignment", imagePreprocess.LetterboxAlignment);
        writer.WriteNumber("targetWidth", imagePreprocess.TargetWidth);
        writer.WriteNumber("targetHeight", imagePreprocess.TargetHeight);
        writer.WriteNumber("resizedWidth", imagePreprocess.ResizedWidth);
        writer.WriteNumber("resizedHeight", imagePreprocess.ResizedHeight);
        writer.WriteNumber("padX", imagePreprocess.PadX);
        writer.WriteNumber("padY", imagePreprocess.PadY);
        writer.WriteNumber("scaleX", imagePreprocess.ResizeScaleX);
        writer.WriteNumber("scaleY", imagePreprocess.ResizeScaleY);
        writer.WriteNumber("fillValue", imagePreprocess.FillValue);
        writer.WriteEndObject();
    }

    private static void WriteEngine(Utf8JsonWriter writer, YoloVisionOutputReportContext context)
    {
        writer.WritePropertyName("engine");
        writer.WriteStartObject();
        writer.WriteString("path", string.Empty);
        writer.WriteString("sha256", string.Empty);
        writer.WritePropertyName("inputShape");
        WriteIntArray(writer, context.InputShape);
        writer.WriteString("precision", "unknown");
        writer.WriteString("materialization", "in-memory-from-onnx");
        writer.WriteString("modelPath", context.ModelPath);
        writer.WriteString("modelSha256", ComputeFileSha256OrEmpty(context.ModelPath));
        writer.WriteNumber("engineDeviceMemoryBytes", context.EngineDeviceMemoryBytes);
        writer.WriteEndObject();
    }

    private static void WriteRuntime(Utf8JsonWriter writer, YoloVisionOutputReportContext context)
    {
        writer.WritePropertyName("runtime");
        writer.WriteStartObject();
        writer.WriteString("os", RuntimeInformation.OSDescription);
        writer.WriteString("rid", RuntimeInformation.RuntimeIdentifier);
        writer.WriteString("cudaVersion", "captured-by-run-log");
        writer.WriteString("tensorRtVersion", $"line-{context.TensorRtLine}");
        writer.WriteString("runtimePackageId", "owner-record-required");
        writer.WriteNumber("tensorRtLine", context.TensorRtLine);
        writer.WriteNumber("profileIndex", context.ProfileIndex);
        writer.WriteNumber("elapsedMilliseconds", context.ElapsedMilliseconds);
        writer.WriteEndObject();
    }

    private static void WriteOutputs(Utf8JsonWriter writer, YoloRuntimeOutputSet outputs)
    {
        writer.WritePropertyName("outputs");
        writer.WriteStartArray();
        foreach (YoloRuntimeOutputTensor output in outputs.Outputs)
        {
            writer.WriteStartObject();
            writer.WriteString("name", output.Name);
            writer.WriteString("role", ToSchemaRole(output.Role));
            writer.WritePropertyName("shape");
            WriteIntArray(writer, output.Shape);
            writer.WriteNumber("elementCount", output.ElementCount);
            writer.WriteString("valueSha256", ComputeFloatSha256(output.Values));
            writer.WritePropertyName("valuePreview");
            WriteFloatPreview(writer, output.Values);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
    }

    private static void WriteBindingMetadata(
        Utf8JsonWriter writer,
        TensorRtEngineBindingReport bindingReport,
        YoloRuntimeOutputSet outputs)
    {
        writer.WritePropertyName("bindingMetadata");
        writer.WriteStartObject();
        writer.WriteString("engineName", bindingReport.EngineName);
        writer.WriteNumber("profileIndex", bindingReport.ProfileIndex);
        writer.WriteBoolean("isReadyForEnqueue", bindingReport.IsReadyForEnqueue);
        writer.WriteString("evidenceKind", "copied-pointer-free-TensorRtEngineBindingReport; not runtime proof");
        writer.WriteBoolean("isRuntimeProof", false);
        writer.WritePropertyName("tensors");
        writer.WriteStartArray();
        foreach (TensorRtEngineTensorBinding binding in bindingReport.Tensors)
        {
            YoloRuntimeOutputTensor? runtimeOutput = outputs.Outputs.FirstOrDefault(
                item => string.Equals(item.Name, binding.Name, StringComparison.Ordinal));
            string semanticRole = runtimeOutput != null
                ? ToSchemaRole(runtimeOutput.Role)
                : binding.IOMode == TensorRtIOMode.Input ? "input" : "unassigned";
            writer.WriteStartObject();
            writer.WriteNumber("index", binding.Index);
            writer.WriteString("name", binding.Name);
            writer.WriteString("ioMode", binding.IOMode.ToString());
            writer.WriteString("semanticRole", semanticRole);
            writer.WriteString("dataType", binding.DataType.ToString());
            WriteDimsProperty(writer, "engineShape", binding.EngineShape);
            writer.WriteString("location", binding.Location.ToString());
            writer.WriteBoolean("isShapeInferenceIO", binding.IsShapeInferenceIO);
            writer.WriteNumber("bytesPerComponent", binding.BytesPerComponent);
            writer.WriteNumber("componentsPerElement", binding.ComponentsPerElement);
            writer.WriteNumber("effectiveBytesPerComponent", binding.EffectiveBytesPerComponent);
            writer.WriteNumber("effectiveComponentsPerElement", binding.EffectiveComponentsPerElement);
            writer.WriteBoolean("usesDataTypeSizeFallback", binding.UsesDataTypeSizeFallback);
            writer.WriteString("format", binding.Format.ToString());
            writer.WriteString("formatDescription", binding.FormatDescription);
            writer.WriteNumber("vectorizedDimension", binding.VectorizedDimension);
            writer.WriteNumber("profileIndex", binding.ProfileIndex);
            WriteOptionalDimsProperty(writer, "profileMinShape", binding.ProfileMinShape);
            WriteOptionalDimsProperty(writer, "profileOptShape", binding.ProfileOptShape);
            WriteOptionalDimsProperty(writer, "profileMaxShape", binding.ProfileMaxShape);
            writer.WriteBoolean("valueCaptured", runtimeOutput != null);
            if (runtimeOutput != null)
            {
                WriteIntArrayProperty(writer, "runtimeShape", runtimeOutput.Shape);
            }

            writer.WritePropertyName("diagnostics");
            writer.WriteStartArray();
            foreach (string diagnostic in binding.Diagnostics ?? Array.Empty<string>())
            {
                writer.WriteStringValue(diagnostic);
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    private static void WriteDimsProperty(Utf8JsonWriter writer, string propertyName, TensorRtDims dims)
    {
        writer.WritePropertyName(propertyName);
        WriteIntArray(writer, dims.Values);
    }

    private static void WriteOptionalDimsProperty(Utf8JsonWriter writer, string propertyName, TensorRtDims? dims)
    {
        writer.WritePropertyName(propertyName);
        if (dims == null)
        {
            writer.WriteNullValue();
            return;
        }

        WriteIntArray(writer, dims.Values);
    }

    private static void WriteIntArrayProperty(Utf8JsonWriter writer, string propertyName, IReadOnlyList<int> values)
    {
        writer.WritePropertyName(propertyName);
        WriteIntArray(writer, values);
    }

    private static void WritePostprocess(Utf8JsonWriter writer, YoloModelProfile profile)
    {
        writer.WritePropertyName("postprocess");
        writer.WriteStartObject();
        writer.WriteNumber("confidenceThreshold", profile.Postprocess.ConfidenceThreshold);
        writer.WriteNumber("nmsThreshold", profile.Postprocess.IouThreshold);
        writer.WriteNumber("topK", profile.Postprocess.TopK);
        writer.WriteBoolean("applyNms", profile.Postprocess.ApplyNms);
        writer.WriteString("nmsMode", profile.Postprocess.NmsMode.ToString());
        writer.WriteString("layout", profile.Postprocess.Layout.ToString());
        writer.WriteString("angleUnit", "unknown");
        writer.WriteString("angleRange", "owner-record-required");
        writer.WriteEndObject();
    }

    private static void WritePredictions(
        Utf8JsonWriter writer,
        YoloVisionOutputReportContext context,
        YoloVisionResult result,
        IReadOnlyList<string> labels)
    {
        writer.WritePropertyName("predictions");
        writer.WriteStartArray();

        foreach (YoloClassificationPrediction classification in result.Classifications)
        {
            writer.WriteStartObject();
            writer.WriteString("task", "cls");
            writer.WriteNumber("classId", classification.ClassIndex);
            writer.WriteString("className", LabelOrIndex(labels, classification.ClassIndex));
            writer.WriteNumber("score", classification.Score);
            writer.WriteEndObject();
        }

        foreach (YoloDetection detection in result.Detections)
        {
            if (result.TaskType != YoloTaskType.Detection)
            {
                continue;
            }

            writer.WriteStartObject();
            writer.WriteString("task", "det");
            WriteBox(writer, detection);
            writer.WriteNumber("classId", detection.ClassIndex);
            writer.WriteString("className", LabelOrIndex(labels, detection.ClassIndex));
            writer.WriteNumber("score", detection.Score);
            writer.WriteNumber("sourceIndex", detection.SourceIndex);
            writer.WriteEndObject();
        }

        foreach (YoloSegmentationPrediction segmentation in result.Segmentations)
        {
            writer.WriteStartObject();
            writer.WriteString("task", "seg");
            WriteBox(writer, segmentation.Detection);
            writer.WriteNumber("classId", segmentation.Detection.ClassIndex);
            writer.WriteString("className", LabelOrIndex(labels, segmentation.Detection.ClassIndex));
            writer.WriteNumber("score", segmentation.Detection.Score);
            writer.WritePropertyName("maskShape");
            WriteIntArray(writer, new[] { segmentation.Mask.Height, segmentation.Mask.Width });
            writer.WriteNumber("maskPixelCount", segmentation.Mask.CountPixelsAtOrAboveThreshold());
            writer.WriteNumber("maskTotalPixelCount", segmentation.Mask.Values.Length);
            writer.WriteNumber("maskThreshold", segmentation.Mask.Threshold);
            writer.WriteString(
                "maskValueKind",
                segmentation.Mask.ValueKind == YoloSegmentationMaskValueKind.Probability ? "probability" : "raw-logits");
            writer.WriteString("maskPixelCountScope", "prototype-grid-before-crop-resize");
            if (context.ImagePreprocess != null && context.SegmentationSpatialTransform != null)
            {
                WriteSegmentationSpatialTransform(
                    writer,
                    YoloSegmentationSpatialTransform.Apply(
                        segmentation,
                        context.ImagePreprocess,
                        context.SegmentationSpatialTransform));
            }

            writer.WriteEndObject();
        }

        foreach (YoloPosePrediction pose in result.Poses)
        {
            writer.WriteStartObject();
            writer.WriteString("task", "pose");
            WriteBox(writer, pose.Detection);
            writer.WriteNumber("classId", pose.Detection.ClassIndex);
            writer.WriteString("className", LabelOrIndex(labels, pose.Detection.ClassIndex));
            writer.WriteNumber("score", pose.Detection.Score);
            writer.WritePropertyName("keypoints");
            writer.WriteStartArray();
            for (int index = 0; index < pose.Keypoints.Length; index++)
            {
                YoloPoseKeypoint keypoint = pose.Keypoints[index];
                writer.WriteStartObject();
                writer.WriteNumber("index", index);
                writer.WriteNumber("x", keypoint.X);
                writer.WriteNumber("y", keypoint.Y);
                writer.WriteNumber("score", keypoint.Score);
                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        foreach (YoloObbDetection orientedBox in result.OrientedBoxes)
        {
            writer.WriteStartObject();
            writer.WriteString("task", "obb");
            writer.WritePropertyName("center");
            writer.WriteStartObject();
            writer.WriteNumber("x", orientedBox.Box.CenterX);
            writer.WriteNumber("y", orientedBox.Box.CenterY);
            writer.WriteEndObject();
            writer.WritePropertyName("size");
            writer.WriteStartObject();
            writer.WriteNumber("width", orientedBox.Box.Width);
            writer.WriteNumber("height", orientedBox.Box.Height);
            writer.WriteEndObject();
            writer.WriteNumber("angle", orientedBox.AngleRadians);
            writer.WriteString("angleUnit", "radian");
            writer.WriteString("angleRange", "owner-record-required");
            writer.WriteNumber("classId", orientedBox.Box.ClassIndex);
            writer.WriteString("className", LabelOrIndex(labels, orientedBox.Box.ClassIndex));
            writer.WriteNumber("score", orientedBox.Box.Score);
            writer.WriteEndObject();
        }

        if (result.SemanticMap != null)
        {
            writer.WriteStartObject();
            writer.WriteString("task", "sem");
            writer.WriteNumber("classCount", result.SemanticMap.ClassCount);
            writer.WriteNumber("width", result.SemanticMap.Width);
            writer.WriteNumber("height", result.SemanticMap.Height);
            writer.WriteNumber("valueCount", result.SemanticMap.Values.Length);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
    }

    private static void WriteSegmentationSpatialTransform(
        Utf8JsonWriter writer,
        YoloSegmentationSpatialTransformResult transform)
    {
        writer.WritePropertyName("spatialTransform");
        writer.WriteStartObject();
        writer.WriteBoolean("applied", true);
        writer.WriteString(
            "coordinateSpace",
            transform.Options.CoordinateSpace == YoloSegmentationCoordinateSpace.Normalized
                ? "normalized"
                : "model-input-pixels");
        writer.WriteBoolean("cropToDetection", transform.Options.CropToDetection);
        writer.WriteString("interpolation", transform.Interpolation);
        writer.WriteNumber("sourceWidth", transform.Preprocess.SourceWidth);
        writer.WriteNumber("sourceHeight", transform.Preprocess.SourceHeight);
        writer.WriteNumber("modelInputWidth", transform.Preprocess.TargetWidth);
        writer.WriteNumber("modelInputHeight", transform.Preprocess.TargetHeight);
        writer.WriteNumber("resizedWidth", transform.Preprocess.ResizedWidth);
        writer.WriteNumber("resizedHeight", transform.Preprocess.ResizedHeight);
        writer.WriteNumber("padX", transform.Preprocess.PadX);
        writer.WriteNumber("padY", transform.Preprocess.PadY);
        writer.WriteNumber("scaleX", transform.EffectiveScaleX);
        writer.WriteNumber("scaleY", transform.EffectiveScaleY);
        writer.WritePropertyName("finalMaskShape");
        WriteIntArray(writer, new[] { transform.Mask.Height, transform.Mask.Width });
        writer.WriteNumber("finalMaskPixelCount", transform.Mask.CountPixelsAtOrAboveThreshold());
        writer.WriteNumber("finalMaskTotalPixelCount", transform.Mask.Values.Length);
        writer.WriteNumber("finalMaskThreshold", transform.Mask.Threshold);
        writer.WriteString("finalMaskValueKind", "probability");
        writer.WriteString("finalMaskScope", transform.Scope);
        writer.WritePropertyName("sourceBox");
        WriteBoxValue(writer, transform.Detection);
        writer.WriteString("boundary", transform.Boundary);
        writer.WriteEndObject();
    }

    private static void WriteBoundary(Utf8JsonWriter writer)
    {
        writer.WritePropertyName("boundary");
        writer.WriteStartObject();
        writer.WriteBoolean("isRuntimeProof", false);
        writer.WriteString("evidenceKind", BoundaryEvidenceKind);
        writer.WritePropertyName("forbiddenSubstitutes");
        writer.WriteStartArray();
        foreach (string item in new[]
        {
            "build-only",
            "dry-run",
            "template",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "readonly diagnostics"
        })
        {
            writer.WriteStringValue(item);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    private static void WriteBox(Utf8JsonWriter writer, YoloDetection detection)
    {
        writer.WritePropertyName("box");
        WriteBoxValue(writer, detection);
    }

    private static void WriteBoxValue(Utf8JsonWriter writer, YoloDetection detection)
    {
        writer.WriteStartObject();
        writer.WriteNumber("x", detection.CenterX);
        writer.WriteNumber("y", detection.CenterY);
        writer.WriteNumber("width", detection.Width);
        writer.WriteNumber("height", detection.Height);
        writer.WriteEndObject();
    }

    private static void WriteIntArray(Utf8JsonWriter writer, IReadOnlyList<int> values)
    {
        writer.WriteStartArray();
        foreach (int value in values)
        {
            writer.WriteNumberValue(value);
        }

        writer.WriteEndArray();
    }

    private static string ComputeFileSha256OrEmpty(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return string.Empty;
        }

        using FileStream stream = File.OpenRead(path);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string ComputeFloatSha256(float[] values)
    {
        if (values == null || values.Length == 0)
        {
            return string.Empty;
        }

        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(values.AsSpan());
        byte[] hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static void WriteFloatPreview(Utf8JsonWriter writer, float[] values)
    {
        writer.WriteStartArray();
        if (values != null)
        {
            int count = Math.Min(values.Length, 8);
            for (int index = 0; index < count; index++)
            {
                writer.WriteNumberValue(values[index]);
            }
        }

        writer.WriteEndArray();
    }

    private static void WriteFloatProperty(Utf8JsonWriter writer, string name, float value)
    {
        if (float.IsFinite(value))
        {
            writer.WriteNumber(name, value);
            return;
        }

        writer.WriteString(name, float.IsNaN(value) ? "NaN" : value > 0.0f ? "Infinity" : "-Infinity");
    }

    private static string LabelOrIndex(IReadOnlyList<string> labels, int index)
    {
        return index >= 0 && index < labels.Count && !string.IsNullOrWhiteSpace(labels[index])
            ? labels[index]
            : index.ToString();
    }

    private static string ToTaskAlias(YoloTaskType taskType)
    {
        return taskType switch
        {
            YoloTaskType.Detection => "det",
            YoloTaskType.Classification => "cls",
            YoloTaskType.Segmentation => "seg",
            YoloTaskType.OrientedBoundingBox => "obb",
            YoloTaskType.Pose => "pose",
            YoloTaskType.SemanticSegmentation => "sem",
            _ => "det"
        };
    }

    private static string ToFamilyAlias(YoloModelFamily family)
    {
        return family switch
        {
            YoloModelFamily.YoloV5 => "yolov5",
            YoloModelFamily.YoloV6 => "yolov6",
            YoloModelFamily.YoloV7 => "yolov7",
            YoloModelFamily.YoloV8 => "yolov8",
            YoloModelFamily.YoloV9 => "yolov9",
            YoloModelFamily.YoloV10 => "yolov10",
            YoloModelFamily.YoloV11 => "yolov11",
            YoloModelFamily.YoloV26 => "yolov26",
            YoloModelFamily.YoloX => "yolox",
            _ => "custom"
        };
    }

    private static string ToSchemaRole(YoloOutputTensorRole role)
    {
        return role switch
        {
            YoloOutputTensorRole.Detection => "boxes",
            YoloOutputTensorRole.Classification => "logits",
            YoloOutputTensorRole.SemanticMap => "semantic",
            YoloOutputTensorRole.MaskPrototypes => "prototypes",
            YoloOutputTensorRole.MaskCoefficients => "maskCoefficients",
            YoloOutputTensorRole.ObbAngles => "obb",
            YoloOutputTensorRole.PoseKeypoints => "keypoints",
            _ => "unknown"
        };
    }
}
