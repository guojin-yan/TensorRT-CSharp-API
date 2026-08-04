using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free copied snapshot of a TensorRT ONNX parser configuration.
/// TensorRT ONNX parser 配置的无指针复制快照。
/// </summary>
public sealed class TensorRtOnnxConfigSnapshot
{
    /// <summary>
    /// Initializes a new copied ONNX config snapshot.
    /// 初始化复制后的 ONNX config 快照。
    /// </summary>
    public TensorRtOnnxConfigSnapshot(
        TensorRtApiLine line,
        TensorRtDataType modelDataType,
        int verbosityLevel,
        string? modelFileName,
        string? textFileName,
        string? fullTextFileName,
        bool printLayerInfo)
    {
        Line = line;
        ModelDataType = modelDataType;
        VerbosityLevel = verbosityLevel;
        ModelFileName = modelFileName ?? string.Empty;
        TextFileName = textFileName ?? string.Empty;
        FullTextFileName = fullTextFileName ?? string.Empty;
        PrintLayerInfo = printLayerInfo;
    }

    /// <summary>
    /// Gets the TensorRT API line that produced the snapshot.
    /// 获取生成该快照的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the copied ONNX model data type.
    /// 获取复制后的 ONNX 模型数据类型。
    /// </summary>
    public TensorRtDataType ModelDataType { get; }

    /// <summary>
    /// Gets the copied parser verbosity level.
    /// 获取复制后的 parser 日志详细级别。
    /// </summary>
    public int VerbosityLevel { get; }

    /// <summary>
    /// Gets the copied ONNX model file name.
    /// 获取复制后的 ONNX model 文件名。
    /// </summary>
    public string ModelFileName { get; }

    /// <summary>
    /// Gets the copied parser text output file name.
    /// 获取复制后的 parser text 输出文件名。
    /// </summary>
    public string TextFileName { get; }

    /// <summary>
    /// Gets the copied parser full text output file name.
    /// 获取复制后的 parser full text 输出文件名。
    /// </summary>
    public string FullTextFileName { get; }

    /// <summary>
    /// Gets whether parser layer information printing is enabled.
    /// 获取是否启用 parser layer 信息打印。
    /// </summary>
    public bool PrintLayerInfo { get; }

    /// <summary>
    /// Creates a compact release-facing summary from this copied snapshot.
    /// 根据该复制快照创建面向发布证据的摘要。
    /// </summary>
    /// <returns>A pointer-free summary. 无指针摘要。</returns>
    public TensorRtOnnxConfigSummary ToSummary()
    {
        return new TensorRtOnnxConfigSummary(
            Line,
            ModelDataType,
            VerbosityLevel,
            PrintLayerInfo,
            ModelFileName.Length,
            TextFileName.Length,
            FullTextFileName.Length);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Line={Line};ModelDataType={ModelDataType};VerbosityLevel={VerbosityLevel};ModelFileNameLength={ModelFileName.Length};TextFileNameLength={TextFileName.Length};FullTextFileNameLength={FullTextFileName.Length};PrintLayerInfo={PrintLayerInfo}";
    }
}
