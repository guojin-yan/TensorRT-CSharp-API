using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
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
    /// Returns whether the selected TensorRT parser reports support for an ONNX operator.
    /// 返回当前 TensorRT parser 是否报告支持指定 ONNX operator。
    /// </summary>
    /// <param name="operatorName">The ONNX operator name, for example <c>Identity</c>. ONNX operator 名称，例如 <c>Identity</c>。</param>
    /// <returns><c>true</c> when TensorRT reports support for the operator. TensorRT 报告支持该 operator 时返回 <c>true</c>。</returns>
    public bool SupportsOperator(string operatorName)
    {
        return NativeBridgeApi.OnnxParserSupportsOperator(Line, _handle, operatorName);
    }

    /// <summary>
    /// Returns whether TensorRT reports the ONNX subgraph at the specified index as supported.
    /// 返回 TensorRT 是否报告指定索引处的 ONNX subgraph 受支持。
    /// </summary>
    /// <param name="index">Zero-based subgraph index. 从零开始的 subgraph 索引。</param>
    /// <returns><c>true</c> when TensorRT reports the subgraph as supported. TensorRT 报告该 subgraph 受支持时返回 <c>true</c>。</returns>
    /// <remarks>
    /// This is a copied scalar query over the native parser state. The native bridge keeps TensorRT version guards in place and does not expose a borrowed parser pointer.
    /// 这是对 native parser 状态的标量只读查询；native bridge 保留 TensorRT 版本保护，不暴露 borrowed parser 指针。
    /// </remarks>
    public bool IsSubgraphSupported(long index)
    {
        return NativeBridgeApi.IsOnnxParserSubgraphSupported(Line, _handle, index);
    }

}
