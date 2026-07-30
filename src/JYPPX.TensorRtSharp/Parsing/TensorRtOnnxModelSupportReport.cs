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
    /// Converts this report into a compact pointer-free summary.
    /// 将当前报告转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads copied managed values that are already present in the report.
    /// It does not call TensorRT, does not expose parser-owned tensor pointers, and does not promote
    /// the report to runtime execution or package-consumer proof.
    /// 此方法只读取报告中已经复制到托管侧的值；不会调用 TensorRT、不会暴露 parser 拥有的 tensor 指针，
    /// 也不会将报告晋级为 runtime execution 或 package-consumer proof。
    /// </remarks>
    /// <returns>A compact ONNX model support summary. 紧凑 ONNX 模型支持性摘要。</returns>
    public TensorRtOnnxModelSupportSummary ToSummary()
    {
        long copiedNodeCount = 0;
        long copiedSupportedSubgraphCount = 0;
        long copiedUnsupportedSubgraphCount = 0;

        for (int index = 0; index < Subgraphs.Count; index++)
        {
            TensorRtOnnxSubgraphSupportInfo subgraph = Subgraphs[index];
            copiedNodeCount += subgraph.Nodes.Count;
            if (subgraph.IsSupported)
            {
                copiedSupportedSubgraphCount++;
            }
            else
            {
                copiedUnsupportedSubgraphCount++;
            }
        }

        return new TensorRtOnnxModelSupportSummary(
            IsSupported,
            SupportedSubgraphCount,
            UnsupportedSubgraphCount,
            Subgraphs.Count,
            copiedSupportedSubgraphCount,
            copiedUnsupportedSubgraphCount,
            copiedNodeCount);
    }

    /// <summary>
    /// Returns a compact support summary.
    /// 返回紧凑的支持性摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Supported={IsSupported} Subgraphs={Subgraphs.Count} SupportedSubgraphs={SupportedSubgraphCount} UnsupportedSubgraphs={UnsupportedSubgraphCount}";
    }
}
