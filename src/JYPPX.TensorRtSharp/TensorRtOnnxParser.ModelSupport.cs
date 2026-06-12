using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParser
{
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
