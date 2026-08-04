using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one diagnostic request for the future TensorRT debug-listener callback owner.
/// 描述未来 TensorRT debug-listener callback owner 的一次诊断请求。
/// </summary>
/// <remarks>
/// This request contains copied metadata only. It does not carry a TensorRT tensor pointer, tensor buffer pointer, CUDA
/// stream handle, or debug tensor data ownership.
/// 该请求只包含复制出的元数据；不携带 TensorRT tensor pointer、tensor buffer pointer、CUDA stream handle 或 debug tensor
/// 数据所有权。
/// </remarks>
public readonly struct TensorRtDebugListenerCallbackRequest
{
    private const int MaxShapeRank = 8;
    private readonly long[] _shapeDimensions;

    /// <summary>
    /// Creates a debug-listener owner diagnostic request.
    /// 创建 debug-listener owner 诊断请求。
    /// </summary>
    /// <param name="tensorName">The copied debug tensor name. 复制出的 debug tensor 名称。</param>
    /// <param name="dataType">The copied debug tensor data type. 复制出的 debug tensor 数据类型。</param>
    /// <param name="location">The copied debug tensor location. 复制出的 debug tensor 位置。</param>
    /// <param name="shapeDimensions">The copied debug tensor shape dimensions. 复制出的 debug tensor shape 维度。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    /// <param name="isInput">Whether the copied metadata describes an input tensor. 复制出的元数据是否表示输入 tensor。</param>
    /// <param name="isOutput">Whether the copied metadata describes an output tensor. 复制出的元数据是否表示输出 tensor。</param>
    /// <param name="isShapeTensor">Whether the copied metadata describes a shape tensor. 复制出的元数据是否表示 shape tensor。</param>
    /// <param name="isExecutionTensor">Whether the copied metadata describes an execution tensor. 复制出的元数据是否表示 execution tensor。</param>
    public TensorRtDebugListenerCallbackRequest(
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        long[]? shapeDimensions,
        string reason = "",
        bool isInput = false,
        bool isOutput = false,
        bool isShapeTensor = false,
        bool isExecutionTensor = true)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Debug listener tensor name must not be empty.", nameof(tensorName));
        }

        if (!Enum.IsDefined(typeof(TensorRtDataType), dataType))
        {
            throw new ArgumentOutOfRangeException(nameof(dataType), "Debug listener data type must be a known TensorRT data type.");
        }

        if (!Enum.IsDefined(typeof(TensorRtTensorLocation), location))
        {
            throw new ArgumentOutOfRangeException(nameof(location), "Debug listener tensor location must be a known TensorRT tensor location.");
        }

        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        if (_shapeDimensions.Length > MaxShapeRank)
        {
            throw new ArgumentOutOfRangeException(nameof(shapeDimensions), "Debug listener diagnostic shape rank must be 8 or less.");
        }

        TensorName = tensorName;
        DataType = dataType;
        Location = location;
        Reason = reason ?? string.Empty;
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
    }

    /// <summary>Gets the copied debug tensor name. 获取复制出的 debug tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 获取复制出的 debug tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 获取复制出的 debug tensor 位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 获取复制出的 debug tensor shape rank。</summary>
    public int ShapeRank => _shapeDimensions.Length;

    /// <summary>Gets the copied debug tensor shape dimensions. 获取复制出的 debug tensor shape 维度。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions);

    /// <summary>Gets the copied diagnostic reason. 获取复制出的诊断原因。</summary>
    public string Reason { get; }

    /// <summary>Gets whether the copied metadata describes an input tensor. 获取复制出的元数据是否表示输入 tensor。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether the copied metadata describes an output tensor. 获取复制出的元数据是否表示输出 tensor。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether the copied metadata describes a shape tensor. 获取复制出的元数据是否表示 shape tensor。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether the copied metadata describes an execution tensor. 获取复制出的元数据是否表示 execution tensor。</summary>
    public bool IsExecutionTensor { get; }

    internal long[] CopyShapeDimensions()
    {
        return (long[])_shapeDimensions.Clone();
    }
}
