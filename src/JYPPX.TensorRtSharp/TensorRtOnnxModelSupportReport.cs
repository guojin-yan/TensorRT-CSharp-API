using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes whether TensorRT reports support for an ONNX model and its parser subgraphs.
/// 描述 TensorRT 是否支持指定 ONNX 模型以及 parser 子图支持情况。
/// </summary>
public sealed class TensorRtOnnxModelSupportReport
{
    /// <summary>
    /// Creates an ONNX model support report.
    /// 创建 ONNX 模型支持性报告。
    /// </summary>
    /// <param name="isSupported">Whether TensorRT reports the full model as supported. TensorRT 是否报告完整模型受支持。</param>
    /// <param name="supportedSubgraphCount">Number of supported subgraphs. 受支持子图数量。</param>
    /// <param name="unsupportedSubgraphCount">Number of unsupported subgraphs. 不受支持子图数量。</param>
    /// <param name="subgraphs">Per-subgraph support details. 每个子图的支持性详情。</param>
    public TensorRtOnnxModelSupportReport(
        bool isSupported,
        long supportedSubgraphCount,
        long unsupportedSubgraphCount,
        IReadOnlyList<TensorRtOnnxSubgraphSupportInfo> subgraphs)
    {
        IsSupported = isSupported;
        SupportedSubgraphCount = supportedSubgraphCount;
        UnsupportedSubgraphCount = unsupportedSubgraphCount;
        Subgraphs = subgraphs ?? Array.Empty<TensorRtOnnxSubgraphSupportInfo>();
    }

    /// <summary>
    /// Gets whether TensorRT reports the whole ONNX model as supported.
    /// 获取 TensorRT 是否报告整个 ONNX 模型受支持。
    /// </summary>
    public bool IsSupported { get; }

    /// <summary>
    /// Gets the number of supported subgraphs reported by TensorRT.
    /// 获取 TensorRT 报告的受支持子图数量。
    /// </summary>
    public long SupportedSubgraphCount { get; }

    /// <summary>
    /// Gets the number of unsupported subgraphs reported by TensorRT.
    /// 获取 TensorRT 报告的不受支持子图数量。
    /// </summary>
    public long UnsupportedSubgraphCount { get; }

    /// <summary>
    /// Gets the per-subgraph support details.
    /// 获取每个子图的支持性详情。
    /// </summary>
    public IReadOnlyList<TensorRtOnnxSubgraphSupportInfo> Subgraphs { get; }

    /// <summary>
    /// Returns a compact support summary.
    /// 返回紧凑的支持性摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Supported={IsSupported} Subgraphs={Subgraphs.Count} SupportedSubgraphs={SupportedSubgraphCount} UnsupportedSubgraphs={UnsupportedSubgraphCount}";
    }
}

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
