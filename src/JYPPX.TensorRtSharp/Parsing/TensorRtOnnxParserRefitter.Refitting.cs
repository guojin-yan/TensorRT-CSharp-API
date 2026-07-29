using System;
using System.IO;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParserRefitter
{
    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes.
    /// 使用已序列化 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The model buffer is pinned only for the native call; TensorRT parser-refitter diagnostics remain available through copied diagnostic APIs.
    /// 模型缓冲区仅在 native 调用期间短期 pin；TensorRT parser-refitter 诊断仍通过复制型诊断 API 获取。
    /// </remarks>
    public bool RefitFromBytes(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.RefitOnnxParserRefitterFromBytes(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Refits the target engine from a serialized ONNX model byte-array segment.
    /// 使用托管字节数组片段中的 ONNX 模型重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model byte segment. 已序列化 ONNX 模型字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return RefitFromBytes(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes in a read-only span.
    /// 使用只读 span 中的 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return RefitFromBytes(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Refits the target engine from serialized ONNX model bytes read from a stream.
    /// 使用从 stream 读取的 ONNX 模型字节重整目标 engine。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromBytes(Stream modelStream, string? modelPath = null)
    {
        return RefitFromBytes(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Refits the target engine from an ONNX model file.
    /// 使用 ONNX 模型文件重整目标 engine。
    /// </summary>
    /// <param name="filePath">The ONNX model path. ONNX 模型路径。</param>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", filePath);
        }

        return NativeBridgeApi.RefitOnnxParserRefitterFromFile(Line, _handle, filePath);
    }
}
