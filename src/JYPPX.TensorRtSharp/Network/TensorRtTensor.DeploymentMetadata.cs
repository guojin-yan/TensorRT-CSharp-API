using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtTensor
{
    /// <summary>
    /// Gets whether TensorRT reports this network tensor as a shape tensor.
    /// 获取 TensorRT 是否将当前网络张量识别为 shape tensor。
    /// </summary>
    /// <remarks>
    /// The result is most reliable after network construction is complete.
    /// 该结果在网络构建完成后最可靠。
    /// </remarks>
    public bool IsShapeTensor => NativeBridgeApi.IsTensorShapeTensor(Line, _handle);

    /// <summary>
    /// Gets whether TensorRT reports this network tensor as an execution tensor.
    /// 获取 TensorRT 是否将当前网络张量识别为 execution tensor。
    /// </summary>
    /// <remarks>
    /// This query helps diagnose dynamic-shape graphs where a tensor may participate only in shape calculation.
    /// 该查询可用于诊断动态 shape 图中张量是否只参与 shape 计算。
    /// </remarks>
    public bool IsExecutionTensor => NativeBridgeApi.IsTensorExecutionTensor(Line, _handle);

    /// <summary>
    /// Gets whether TensorRT reports this tensor as a network input tensor.
    /// 获取 TensorRT 是否将当前张量报告为 network input tensor。
    /// </summary>
    /// <remarks>
    /// This query is useful when validating direct network construction and layer wiring.
    /// 该查询适合验证直接构图和 layer 连接关系。
    /// </remarks>
    public bool IsNetworkInput => NativeBridgeApi.IsTensorNetworkInput(Line, _handle);

    /// <summary>
    /// Gets whether TensorRT reports this tensor as a network output tensor.
    /// 获取 TensorRT 是否将当前张量报告为 network output tensor。
    /// </summary>
    /// <remarks>
    /// Marked network outputs should report <c>true</c> after the network output has been configured.
    /// 完成输出标记后，network output 通常应返回 <c>true</c>。
    /// </remarks>
    public bool IsNetworkOutput => NativeBridgeApi.IsTensorNetworkOutput(Line, _handle);

    /// <summary>
    /// Gets the symbolic name assigned to one tensor dimension.
    /// 获取指定张量维度的符号名称。
    /// </summary>
    /// <param name="dimensionIndex">The zero-based tensor dimension index. 从零开始的张量维度索引。</param>
    /// <returns>The symbolic dimension name, or an empty string when no name is assigned. 返回维度符号名；未设置时返回空字符串。</returns>
    public string GetDimensionName(int dimensionIndex)
    {
        ValidateDimensionIndex(dimensionIndex);
        return NativeBridgeApi.GetTensorDimensionName(Line, _handle, dimensionIndex);
    }

    /// <summary>
    /// Assigns a symbolic name to one tensor dimension.
    /// 为指定张量维度设置符号名称。
    /// </summary>
    /// <param name="dimensionIndex">The zero-based tensor dimension index. 从零开始的张量维度索引。</param>
    /// <param name="name">The symbolic dimension name. 维度符号名称。</param>
    /// <remarks>
    /// Dimension names can express runtime equality constraints and improve diagnostic readability in dynamic-shape networks.
    /// 维度名称可表达运行时维度相等约束，并提升动态 shape 网络的诊断可读性。
    /// </remarks>
    public void SetDimensionName(int dimensionIndex, string name)
    {
        ValidateDimensionIndex(dimensionIndex);
        NativeBridgeApi.SetTensorDimensionName(Line, _handle, dimensionIndex, name);
    }

    /// <summary>
    /// Clears the symbolic name assigned to one tensor dimension.
    /// 清除指定张量维度的符号名称。
    /// </summary>
    /// <param name="dimensionIndex">The zero-based tensor dimension index. 从零开始的张量维度索引。</param>
    public void ClearDimensionName(int dimensionIndex)
    {
        ValidateDimensionIndex(dimensionIndex);
        NativeBridgeApi.ClearTensorDimensionName(Line, _handle, dimensionIndex);
    }

    private static void ValidateDimensionIndex(int dimensionIndex)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }
    }
}
