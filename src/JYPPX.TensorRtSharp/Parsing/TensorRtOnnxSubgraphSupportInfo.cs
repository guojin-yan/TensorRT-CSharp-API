using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one ONNX parser subgraph support result.
/// 描述一条 ONNX parser 子图支持性结果。
/// </summary>
public sealed class TensorRtOnnxSubgraphSupportInfo
{
    /// <summary>
    /// Creates a subgraph support result.
    /// 创建子图支持性结果。
    /// </summary>
    /// <param name="index">Zero-based subgraph index. 从零开始的子图索引。</param>
    /// <param name="isSupported">Whether TensorRT reports the subgraph as supported. TensorRT 是否报告该子图受支持。</param>
    /// <param name="nodes">ONNX node indexes included in this subgraph. 当前子图包含的 ONNX 节点索引。</param>
    public TensorRtOnnxSubgraphSupportInfo(long index, bool isSupported, IReadOnlyList<long> nodes)
    {
        Index = index;
        IsSupported = isSupported;
        Nodes = nodes ?? Array.Empty<long>();
    }

    /// <summary>
    /// Gets the zero-based subgraph index.
    /// 获取从零开始的子图索引。
    /// </summary>
    public long Index { get; }

    /// <summary>
    /// Gets whether TensorRT reports this subgraph as supported.
    /// 获取 TensorRT 是否报告该子图受支持。
    /// </summary>
    public bool IsSupported { get; }

    /// <summary>
    /// Gets the ONNX node indexes included in this subgraph.
    /// 获取当前子图包含的 ONNX 节点索引。
    /// </summary>
    public IReadOnlyList<long> Nodes { get; }

    /// <summary>
    /// Returns a compact subgraph summary.
    /// 返回紧凑的子图摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Index={Index} Supported={IsSupported} Nodes={Nodes.Count}";
    }
}
