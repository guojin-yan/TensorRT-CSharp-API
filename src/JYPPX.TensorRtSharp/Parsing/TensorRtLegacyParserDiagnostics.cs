using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Provides pointer-free readonly diagnostics for TensorRT 8 legacy UFF and Caffe parser assets.
/// 提供 TensorRT 8 legacy UFF 与 Caffe parser 资产的无指针只读诊断。
/// </summary>
public static class TensorRtLegacyParserDiagnostics
{
    /// <summary>
    /// Gets the UFF version required by the TensorRT 8 legacy parser.
    /// 获取 TensorRT 8 legacy parser 所需的 UFF 版本。
    /// </summary>
    /// <param name="line">The TensorRT adapter line; only TensorRT 8 is supported. TensorRT 适配版本线；仅支持 TensorRT 8。</param>
    /// <returns>A caller-owned version snapshot. caller-owned 版本快照。</returns>
    public static TensorRtLegacyUffRequiredVersionSnapshot GetUffRequiredVersion(
        TensorRtApiLine line = TensorRtApiLine.TensorRt8)
    {
        return Internal.Interop.NativeBridgeApi.GetLegacyUffRequiredVersion(line);
    }

    /// <summary>
    /// Parses a Caffe binaryproto file and copies its dimensions, data type, and data bytes.
    /// 解析 Caffe binaryproto 文件并复制其维度、数据类型和数据字节。
    /// </summary>
    /// <param name="filePath">The binaryproto file path. binaryproto 文件路径。</param>
    /// <param name="line">The TensorRT adapter line; only TensorRT 8 is supported. TensorRT 适配版本线；仅支持 TensorRT 8。</param>
    /// <returns>A pointer-free caller-owned snapshot. 无指针的 caller-owned 快照。</returns>
    public static TensorRtCaffeBinaryProtoSnapshot ReadCaffeBinaryProto(
        string filePath,
        TensorRtApiLine line = TensorRtApiLine.TensorRt8)
    {
        return Internal.Interop.NativeBridgeApi.ReadLegacyCaffeBinaryProto(line, filePath);
    }
}
