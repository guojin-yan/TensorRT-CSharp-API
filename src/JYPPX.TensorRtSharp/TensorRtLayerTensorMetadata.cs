using System;
using System.Collections.Generic;
using System.Linq;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one TensorRT layer input or output tensor slot for deployment diagnostics.
/// 描述 TensorRT layer 的一个输入或输出 tensor 槽位，用于部署诊断。
/// </summary>
/// <remarks>
/// This type is a managed snapshot. It does not own or expose the native TensorRT <c>ITensor*</c> pointer.
/// 此类型是托管快照，不持有也不暴露原生 TensorRT <c>ITensor*</c> 指针。
/// </remarks>
public sealed class TensorRtLayerTensorMetadata
{
    internal TensorRtLayerTensorMetadata(
        bool isOutputSlot,
        int index,
        bool hasTensor,
        string name,
        TensorRtDataType? dataType,
        TensorRtDims? shape,
        TensorRtTensorLocation? location,
        TensorRtTensorFormats? allowedFormats,
        bool isShapeTensor,
        bool isExecutionTensor,
        bool isNetworkInput,
        bool isNetworkOutput,
        bool hasDynamicDimension,
        IReadOnlyList<string> dimensionNames,
        IReadOnlyList<int> dimensionExtents,
        string summary,
        TensorRtDims64? shape64 = null,
        IReadOnlyList<long>? dimensionExtents64 = null)
    {
        IsOutputSlot = isOutputSlot;
        Index = index;
        HasTensor = hasTensor;
        Name = name ?? string.Empty;
        DataType = dataType;
        Shape = shape;
        Location = location;
        AllowedFormats = allowedFormats;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        IsNetworkInput = isNetworkInput;
        IsNetworkOutput = isNetworkOutput;
        HasDynamicDimension = hasDynamicDimension;
        DimensionNames = dimensionNames ?? Array.Empty<string>();
        DimensionExtents = dimensionExtents ?? Array.Empty<int>();
        Shape64 = shape64;
        DimensionExtents64 = dimensionExtents64 ?? Array.Empty<long>();
        Summary = summary ?? string.Empty;
    }

    /// <summary>
    /// Gets whether this slot is an output slot.
    /// 获取当前槽位是否为输出槽位。
    /// </summary>
    public bool IsOutputSlot { get; }

    /// <summary>
    /// Gets the zero-based layer input or output slot index.
    /// 获取从零开始的 layer 输入或输出槽位索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets whether the slot currently contains a TensorRT tensor.
    /// 获取当前槽位是否包含 TensorRT tensor。
    /// </summary>
    public bool HasTensor { get; }

    /// <summary>
    /// Gets the TensorRT tensor name, or an empty string when absent.
    /// 获取 TensorRT tensor 名称；不存在时为空字符串。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the TensorRT tensor data type when the slot contains a tensor.
    /// 当槽位包含 tensor 时获取 TensorRT tensor 数据类型。
    /// </summary>
    public TensorRtDataType? DataType { get; }

    /// <summary>
    /// Gets the static or dynamic TensorRT dimensions when the slot contains a tensor.
    /// 当槽位包含 tensor 时获取 TensorRT 静态或动态维度。
    /// </summary>
    public TensorRtDims? Shape { get; }

    /// <summary>
    /// Gets the TensorRT 11 tensor dimensions with 64-bit extents when available.
    /// 获取可用时的 TensorRT 11 张量维度，并保留 64 位 extent。
    /// </summary>
    public TensorRtDims64? Shape64 { get; }

    /// <summary>
    /// Gets the TensorRT tensor memory location when available.
    /// 获取可用时的 TensorRT tensor 内存位置。
    /// </summary>
    public TensorRtTensorLocation? Location { get; }

    /// <summary>
    /// Gets the allowed TensorRT tensor formats when available.
    /// 获取可用时的 TensorRT tensor 允许格式。
    /// </summary>
    public TensorRtTensorFormats? AllowedFormats { get; }

    /// <summary>
    /// Gets whether TensorRT reports the tensor as a shape tensor.
    /// 获取 TensorRT 是否将该 tensor 报告为 shape tensor。
    /// </summary>
    public bool IsShapeTensor { get; }

    /// <summary>
    /// Gets whether TensorRT reports the tensor as an execution tensor.
    /// 获取 TensorRT 是否将该 tensor 报告为 execution tensor。
    /// </summary>
    public bool IsExecutionTensor { get; }

    /// <summary>
    /// Gets whether this tensor is a network input tensor.
    /// 获取该 tensor 是否为 network input tensor。
    /// </summary>
    public bool IsNetworkInput { get; }

    /// <summary>
    /// Gets whether this tensor is a network output tensor.
    /// 获取该 tensor 是否为 network output tensor。
    /// </summary>
    public bool IsNetworkOutput { get; }

    /// <summary>
    /// Gets whether any dimension extent is dynamic, typically represented by a negative value.
    /// 获取是否存在动态维度，通常由负数维度表示。
    /// </summary>
    public bool HasDynamicDimension { get; }

    /// <summary>
    /// Gets symbolic dimension names for each tensor dimension. Empty entries mean unnamed dimensions.
    /// 获取每个 tensor 维度的符号名称；空字符串表示该维度未命名。
    /// </summary>
    public IReadOnlyList<string> DimensionNames { get; }

    /// <summary>
    /// Gets raw dimension extents for each tensor dimension.
    /// 获取每个 tensor 维度的原始 extent。
    /// </summary>
    public IReadOnlyList<int> DimensionExtents { get; }

    /// <summary>
    /// Gets raw 64-bit dimension extents for each tensor dimension when available.
    /// 获取可用时每个 tensor 维度的原始 64 位 extent。
    /// </summary>
    public IReadOnlyList<long> DimensionExtents64 { get; }

    /// <summary>
    /// Gets the compact native diagnostic summary for this slot.
    /// 获取该槽位的原生紧凑诊断摘要。
    /// </summary>
    public string Summary { get; }

    /// <summary>
    /// Returns a compact diagnostic string.
    /// 返回紧凑诊断字符串。
    /// </summary>
    public override string ToString()
    {
        if (!HasTensor)
        {
            return $"{(IsOutputSlot ? "output" : "input")}[{Index}]=null";
        }

        IReadOnlyList<long> dimsSource = DimensionExtents64.Count > 0 ? DimensionExtents64 : DimensionExtents.Select(static value => (long)value).ToArray();
        string dims = dimsSource.Count == 0 ? string.Empty : string.Join(",", dimsSource.Select(static value => value.ToString()));
        return $"{(IsOutputSlot ? "output" : "input")}[{Index}] {Name} {DataType} shape=[{dims}] shapeTensor={IsShapeTensor} execTensor={IsExecutionTensor} networkInput={IsNetworkInput} networkOutput={IsNetworkOutput}";
    }
}
