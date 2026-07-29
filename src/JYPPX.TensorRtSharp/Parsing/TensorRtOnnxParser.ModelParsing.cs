using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Parses ONNX models into a TensorRT network definition.
/// 将 ONNX 模型解析到 TensorRT network definition。
/// </summary>
public sealed partial class TensorRtOnnxParser
{
    /// <summary>
    /// Parses an ONNX model from a file path into the target network.
    /// 从文件路径解析 ONNX 模型到目标 network。
    /// </summary>
    /// <param name="filePath">The ONNX model path. ONNX 模型路径。</param>
    /// <param name="verbosity">TensorRT parser verbosity level. TensorRT parser 日志详细级别。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool ParseFromFile(string filePath, int verbosity = 1)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", filePath);
        }

        return NativeBridgeApi.ParseOnnxFromFile(Line, _handle, filePath, verbosity);
    }

    /// <summary>
    /// Parses ONNX model bytes into the target network.
    /// 将 ONNX 模型字节解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化的 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool Parse(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.ParseOnnxFromMemory(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Parses serialized ONNX model bytes through TensorRT's legacy weight-descriptor entry.
    /// 通过 TensorRT legacy weight-descriptor 入口解析已序列化 ONNX 模型字节。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes copied and pinned only for the vendor call. 仅在 vendor 调用期间复制并 pin 的 ONNX 模型字节。</param>
    /// <returns><see langword="true"/> when TensorRT reports a successful parse. TensorRT 报告解析成功时返回 <see langword="true"/>。</returns>
    /// <exception cref="BridgeProbeException">Thrown for TensorRT 11, where this vendor method was removed. TensorRT 11 已移除该 vendor 方法。</exception>
    public bool ParseWithWeightDescriptors(byte[] modelData)
    {
        return NativeBridgeApi.ParseOnnxWithWeightDescriptors(Line, _handle, modelData);
    }

    /// <summary>
    /// Parses ONNX model bytes from a managed byte-array segment into the target network.
    /// 将托管字节数组片段中的 ONNX 模型解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model byte segment. 已序列化 ONNX 模型字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The segment is copied into an exact managed byte array before native interop.
    /// 调用 native interop 前会将片段复制为精确长度的托管字节数组。
    /// </remarks>
    public bool Parse(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return Parse(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Parses ONNX model bytes from a managed read-only span into the target network.
    /// 将托管只读 span 中的 ONNX 模型字节解析到目标 network。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The span is copied into a managed byte array before native interop, and TensorRT does not retain caller-owned memory.
    /// 调用 native interop 前会将 span 复制到托管字节数组，TensorRT 不会保留调用方拥有的内存。
    /// </remarks>
    public bool Parse(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return Parse(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Parses ONNX model bytes from a managed stream into the target network.
    /// 从托管 stream 读取 ONNX 模型字节并解析到目标 network。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The stream is copied into managed memory before native interop; this is not a TensorRT model-proto or external-initializer lifetime bridge.
    /// 调用 native interop 前会将 stream 内容复制到托管内存；该入口不是 TensorRT model proto 或 external initializer 生命周期桥。
    /// </remarks>
    public bool Parse(Stream modelStream, string? modelPath = null)
    {
        return Parse(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

}
