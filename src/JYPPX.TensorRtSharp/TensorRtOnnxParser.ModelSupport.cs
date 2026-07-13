using System;
using System.Collections.Generic;
using System.IO;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParser
{
    /// <summary>
    /// Captures copied ONNX parser diagnostics and plugin-library inventory without exposing native parser-owned pointers.
    /// 捕获已复制的 ONNX parser 诊断和 plugin-library inventory，不暴露原生 parser 拥有的指针。
    /// </summary>
    /// <returns>A pointer-free parser diagnostic snapshot. 无指针逃逸的 parser 诊断快照。</returns>
    public TensorRtOnnxParserDiagnosticSnapshot GetDiagnosticSnapshot()
    {
        IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics = GetDiagnostics();
        IReadOnlyList<string> usedVCPluginLibraries = GetUsedVCPluginLibraries();
        string diagnosticSummary = diagnostics.Count == 0
            ? "ONNX parser reported no errors."
            : BuildDiagnosticSummary(diagnostics);
        bool identityOperatorSupported = SupportsOperator("Identity");

        return new TensorRtOnnxParserDiagnosticSnapshot(
            Line,
            ErrorCount,
            diagnostics,
            diagnosticSummary,
            usedVCPluginLibraries,
            identityOperatorSupported);
    }

    /// <summary>
    /// Gets VC plugin libraries used by the most recent ONNX parser operation.
    /// 获取最近一次 ONNX parser 操作使用的 VC plugin library 列表。
    /// </summary>
    /// <returns>Parser-owned library paths copied into managed strings. 已复制到托管字符串中的 parser library 路径。</returns>
    public IReadOnlyList<string> GetUsedVCPluginLibraries()
    {
        return NativeBridgeApi.GetOnnxParserUsedVCPluginLibraries(Line, _handle);
    }

    /// <summary>
    /// Checks whether TensorRT reports support for the serialized ONNX model.
    /// 检查 TensorRT 是否报告支持指定的已序列化 ONNX 模型。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化的 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns>A support report containing the full-model result and parser subgraphs. 包含完整模型结果和 parser 子图的支持性报告。</returns>
    public TensorRtOnnxModelSupportReport CheckModelSupport(byte[] modelData, string? modelPath = null)
    {
        bool supported = NativeBridgeApi.OnnxParserSupportsModelV2(Line, _handle, modelData, modelPath);
        if (Line == TensorRtApiLine.TensorRt8)
        {
            return new TensorRtOnnxModelSupportReport(supported, 0, 0, Array.Empty<TensorRtOnnxSubgraphSupportInfo>());
        }

        long subgraphCount = NativeBridgeApi.GetOnnxParserSubgraphCount(Line, _handle);
        long supportedSubgraphCount = NativeBridgeApi.GetOnnxParserSupportedSubgraphCount(Line, _handle);
        long unsupportedSubgraphCount = NativeBridgeApi.GetOnnxParserUnsupportedSubgraphCount(Line, _handle);

        if (subgraphCount <= 0)
        {
            return new TensorRtOnnxModelSupportReport(supported, supportedSubgraphCount, unsupportedSubgraphCount, Array.Empty<TensorRtOnnxSubgraphSupportInfo>());
        }

        if (subgraphCount > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "ONNX parser subgraph count is too large for a managed list.");
        }

        TensorRtOnnxSubgraphSupportInfo[] subgraphs = new TensorRtOnnxSubgraphSupportInfo[(int)subgraphCount];
        for (long subgraphIndex = 0; subgraphIndex < subgraphCount; subgraphIndex++)
        {
            bool subgraphSupported = NativeBridgeApi.IsOnnxParserSubgraphSupported(Line, _handle, subgraphIndex);
            long nodeCount = NativeBridgeApi.GetOnnxParserSubgraphNodeCount(Line, _handle, subgraphIndex);
            IReadOnlyList<long> nodes = ReadSubgraphNodes(subgraphIndex, nodeCount);
            subgraphs[(int)subgraphIndex] = new TensorRtOnnxSubgraphSupportInfo(subgraphIndex, subgraphSupported, nodes);
        }

        return new TensorRtOnnxModelSupportReport(supported, supportedSubgraphCount, unsupportedSubgraphCount, subgraphs);
    }

    /// <summary>
    /// Checks whether TensorRT reports support for the serialized ONNX model in a managed byte-array segment.
    /// 检查 TensorRT 是否报告支持托管字节数组片段中的已序列化 ONNX 模型。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model byte segment. 已序列化 ONNX 模型字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns>A support report containing the full-model result and parser subgraphs. 包含完整模型结果和 parser 子图的支持性报告。</returns>
    /// <remarks>
    /// The segment is copied into an exact managed byte array before native interop.
    /// 调用 native interop 前会将片段复制为精确长度的托管字节数组。
    /// </remarks>
    public TensorRtOnnxModelSupportReport CheckModelSupport(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return CheckModelSupport(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Checks whether TensorRT reports support for the serialized ONNX model in a managed read-only span.
    /// 检查 TensorRT 是否报告支持托管只读 span 中的已序列化 ONNX 模型。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model bytes. 已序列化 ONNX 模型字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns>A support report containing the full-model result and parser subgraphs. 包含完整模型结果和 parser 子图的支持性报告。</returns>
    /// <remarks>
    /// The span is copied into a managed byte array before native interop, and TensorRT does not retain caller-owned memory.
    /// 调用 native interop 前会将 span 复制到托管字节数组，TensorRT 不会保留调用方拥有的内存。
    /// </remarks>
    public TensorRtOnnxModelSupportReport CheckModelSupport(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return CheckModelSupport(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Checks whether TensorRT reports support for the serialized ONNX model in a managed stream.
    /// 检查 TensorRT 是否报告支持托管 stream 中的已序列化 ONNX 模型。
    /// </summary>
    /// <param name="modelStream">The readable stream containing serialized ONNX model bytes. 包含已序列化 ONNX 模型字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns>A support report containing the full-model result and parser subgraphs. 包含完整模型结果和 parser 子图的支持性报告。</returns>
    /// <remarks>
    /// The stream is copied into managed memory before native interop; this does not use TensorRT model-proto ownership APIs.
    /// 调用 native interop 前会将 stream 内容复制到托管内存；该入口不使用 TensorRT model proto ownership API。
    /// </remarks>
    public TensorRtOnnxModelSupportReport CheckModelSupport(Stream modelStream, string? modelPath = null)
    {
        return CheckModelSupport(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Returns whether a parsed ONNX layer output tensor exists without exposing the native tensor pointer.
    /// 判断已解析 ONNX layer 的输出 tensor 是否存在，同时不暴露原生 tensor 指针。
    /// </summary>
    /// <param name="layerName">The ONNX layer name to query. 要查询的 ONNX layer 名称。</param>
    /// <param name="outputIndex">Zero-based output index. 从零开始的输出索引。</param>
    /// <returns><see langword="true"/> when TensorRT returns a non-null tensor pointer. TensorRT 返回非空 tensor 指针时为 <see langword="true"/>。</returns>
    public bool LayerOutputTensorExists(string layerName, long outputIndex = 0)
    {
        return NativeBridgeApi.OnnxParserLayerOutputTensorExists(Line, _handle, layerName, outputIndex);
    }

    private IReadOnlyList<long> ReadSubgraphNodes(long subgraphIndex, long nodeCount)
    {
        if (nodeCount <= 0)
        {
            return Array.Empty<long>();
        }

        if (nodeCount > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "ONNX parser subgraph node count is too large for a managed list.");
        }

        long[] nodes = new long[(int)nodeCount];
        for (long nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
        {
            nodes[(int)nodeIndex] = NativeBridgeApi.GetOnnxParserSubgraphNode(Line, _handle, subgraphIndex, nodeIndex);
        }

        return nodes;
    }
}
