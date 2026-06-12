using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes TensorRT weights metadata without exposing the native weights pointer.
/// 描述 TensorRT weights 的元数据，但不暴露原生 weights 指针。
/// </summary>
public sealed class TensorRtWeightsInfo
{
    internal TensorRtWeightsInfo(NativeTensorRtWeightsInfo native)
    {
        DataType = (TensorRtDataType)native.DataType;
        ElementCount = native.Count;
        HasValues = native.HasValues != 0;
    }

    /// <summary>
    /// Gets the TensorRT data type for the weights.
    /// 获取 weights 使用的 TensorRT 数据类型。
    /// </summary>
    public TensorRtDataType DataType { get; }

    /// <summary>
    /// Gets the number of elements in the weights.
    /// 获取 weights 中的元素数量。
    /// </summary>
    public long ElementCount { get; }

    /// <summary>
    /// Gets whether TensorRT returned a non-null native weights value pointer.
    /// 获取 TensorRT 是否返回了非空的原生 weights value 指针。
    /// </summary>
    public bool HasValues { get; }
}
