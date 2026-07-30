using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Compact pointer-free summary of a TensorRT ONNX parser configuration snapshot.
/// TensorRT ONNX parser 配置快照的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtOnnxConfigSummary
{
    /// <summary>
    /// Initializes a compact ONNX config summary.
    /// 初始化紧凑 ONNX config 摘要。
    /// </summary>
    public TensorRtOnnxConfigSummary(
        TensorRtApiLine line,
        TensorRtDataType modelDataType,
        int verbosityLevel,
        bool printLayerInfo,
        int modelFileNameLength,
        int textFileNameLength,
        int fullTextFileNameLength)
    {
        Line = line;
        ModelDataType = modelDataType;
        VerbosityLevel = verbosityLevel;
        PrintLayerInfo = printLayerInfo;
        ModelFileNameLength = modelFileNameLength;
        TextFileNameLength = textFileNameLength;
        FullTextFileNameLength = fullTextFileNameLength;
    }

    /// <summary>
    /// Gets the TensorRT API line that produced the summary.
    /// 获取生成摘要的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the copied model data type.
    /// 获取复制后的模型数据类型。
    /// </summary>
    public TensorRtDataType ModelDataType { get; }

    /// <summary>
    /// Gets the copied parser verbosity level.
    /// 获取复制后的 parser 日志详细级别。
    /// </summary>
    public int VerbosityLevel { get; }

    /// <summary>
    /// Gets whether parser layer information printing is enabled.
    /// 获取是否启用 parser layer 信息打印。
    /// </summary>
    public bool PrintLayerInfo { get; }

    /// <summary>
    /// Gets whether a model file name is present without exposing the full path.
    /// 获取是否存在 model 文件名，同时不暴露完整路径。
    /// </summary>
    public bool HasModelFileName => ModelFileNameLength > 0;

    /// <summary>
    /// Gets whether a text output file name is present without exposing the full path.
    /// 获取是否存在 text 输出文件名，同时不暴露完整路径。
    /// </summary>
    public bool HasTextFileName => TextFileNameLength > 0;

    /// <summary>
    /// Gets whether a full text output file name is present without exposing the full path.
    /// 获取是否存在 full text 输出文件名，同时不暴露完整路径。
    /// </summary>
    public bool HasFullTextFileName => FullTextFileNameLength > 0;

    /// <summary>
    /// Gets the copied model file name length.
    /// 获取复制后的 model 文件名长度。
    /// </summary>
    public int ModelFileNameLength { get; }

    /// <summary>
    /// Gets the copied text file name length.
    /// 获取复制后的 text 文件名长度。
    /// </summary>
    public int TextFileNameLength { get; }

    /// <summary>
    /// Gets the copied full text file name length.
    /// 获取复制后的 full text 文件名长度。
    /// </summary>
    public int FullTextFileNameLength { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Line={Line};ModelDataType={ModelDataType};VerbosityLevel={VerbosityLevel};PrintLayerInfo={PrintLayerInfo};HasModelFileName={HasModelFileName};HasTextFileName={HasTextFileName};HasFullTextFileName={HasFullTextFileName};ModelFileNameLength={ModelFileNameLength};TextFileNameLength={TextFileNameLength};FullTextFileNameLength={FullTextFileNameLength}";
    }
}
