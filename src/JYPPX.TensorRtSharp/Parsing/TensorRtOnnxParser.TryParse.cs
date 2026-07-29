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
    /// Attempts to parse ONNX model bytes and returns copied TensorRT parser diagnostics.
    /// 尝试解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(byte[] modelData, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        return TryParse(modelData, null, out diagnostics);
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes and returns copied TensorRT parser diagnostics.
    /// 尝试解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(byte[] modelData, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        bool parsed = Parse(modelData, modelPath);
        diagnostics = GetDiagnostics();
        return parsed;
    }

    /// <summary>
    /// Attempts the legacy weight-descriptor parse and returns copied diagnostics.
    /// 尝试 legacy weight-descriptor 解析并返回已复制诊断。
    /// </summary>
    public bool TryParseWithWeightDescriptors(
        byte[] modelData,
        out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        bool parsed = ParseWithWeightDescriptors(modelData);
        diagnostics = GetDiagnostics();
        return parsed;
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes from a managed stream and returns copied TensorRT parser diagnostics.
    /// 尝试从托管 stream 解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(Stream modelStream, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        return TryParse(modelStream, null, out diagnostics);
    }

    /// <summary>
    /// Attempts to parse ONNX model bytes from a managed stream and returns copied TensorRT parser diagnostics.
    /// 尝试从托管 stream 解析 ONNX 模型字节并返回已复制的 TensorRT parser 诊断信息。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <param name="diagnostics">Copied parser diagnostics collected after the parse attempt. 解析尝试后收集到的 parser 诊断信息副本。</param>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool TryParse(Stream modelStream, string? modelPath, out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics)
    {
        bool parsed = Parse(modelStream, modelPath);
        diagnostics = GetDiagnostics();
        return parsed;
    }

}
