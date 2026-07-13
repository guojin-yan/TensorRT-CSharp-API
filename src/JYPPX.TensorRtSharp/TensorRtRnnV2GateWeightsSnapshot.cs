using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a copied TensorRT 8 RNNv2 gate weight or bias snapshot.
/// 表示复制后的 TensorRT 8 RNNv2 gate weight 或 bias 快照。
/// </summary>
public sealed class TensorRtRnnV2GateWeightsSnapshot
{
    private readonly byte[] _values;

    internal TensorRtRnnV2GateWeightsSnapshot(
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights,
        bool isBias,
        TensorRtDataType dataType,
        long elementCount,
        byte[] values)
    {
        LayerIndex = layerIndex;
        Gate = gate;
        IsInputWeights = isInputWeights;
        IsBias = isBias;
        DataType = dataType;
        ElementCount = elementCount;
        _values = values == null
            ? throw new ArgumentNullException(nameof(values))
            : (byte[])values.Clone();
    }

    /// <summary>Gets the physical RNN layer index. 获取 RNN 物理层索引。</summary>
    public int LayerIndex { get; }

    /// <summary>Gets the selected RNN gate. 获取选中的 RNN gate。</summary>
    public TensorRtRnnGateType Gate { get; }

    /// <summary>
    /// Gets whether the snapshot selects W/Wb input-side parameters instead of R/Rb recurrent parameters.
    /// 获取快照是否选择 W/Wb 输入侧参数，而不是 R/Rb recurrent 参数。
    /// </summary>
    public bool IsInputWeights { get; }

    /// <summary>Gets whether this snapshot contains bias values. 获取快照是否包含 bias 值。</summary>
    public bool IsBias { get; }

    /// <summary>Gets the TensorRT element data type. 获取 TensorRT 元素数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the number of elements. 获取元素数量。</summary>
    public long ElementCount { get; }

    /// <summary>Gets the copied byte count. 获取已复制字节数。</summary>
    public int ByteCount => _values.Length;

    /// <summary>
    /// Returns a new copy of the native weight bytes.
    /// 返回 native weight 字节的新副本。
    /// </summary>
    public byte[] ToArray()
    {
        return (byte[])_values.Clone();
    }
}
