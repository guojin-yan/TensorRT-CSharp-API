using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtNetworkDefinition(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets or sets the Line value.
    /// 获取或设置 Line 值。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the Name value.
    /// 获取或设置 Name 值。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetNetworkName(Line, _handle);
        set => NativeBridgeApi.SetNetworkName(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Input Count value.
    /// 获取 Input Count 值。
    /// </summary>
    public int InputCount => NativeBridgeApi.GetNetworkInputCount(Line, _handle);

    /// <summary>
    /// Gets the Output Count value.
    /// 获取 Output Count 值。
    /// </summary>
    public int OutputCount => NativeBridgeApi.GetNetworkOutputCount(Line, _handle);

    /// <summary>
    /// Gets the Layer Count value.
    /// 获取 Layer Count 值。
    /// </summary>
    public int LayerCount => NativeBridgeApi.GetNetworkLayerCount(Line, _handle);

    /// <summary>
    /// Gets the Flags value.
    /// 获取 Flags 值。
    /// </summary>
    public TensorRtNetworkDefinitionCreationFlags Flags => NativeBridgeApi.GetNetworkFlags(Line, _handle);

    /// <summary>
    /// Gets whether this network uses TensorRT implicit batch dimensions.
    /// 获取当前网络是否使用 TensorRT 隐式 batch 维度。
    /// </summary>
    public bool HasImplicitBatchDimension => NativeBridgeApi.HasImplicitBatchDimension(Line, _handle);

    /// <summary>
    /// Gets the Flag value.
    /// 获取 Flag 值。
    /// </summary>
    public bool GetFlag(TensorRtNetworkDefinitionCreationFlags flag)
    {
        return NativeBridgeApi.GetNetworkFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Adds a Input layer or object.
    /// 添加 Input 层或对象。
    /// </summary>
    public TensorRtTensor AddInput(string name, TensorRtDataType dataType, TensorRtDims shape)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.AddNetworkInput(Line, _handle, name, dataType, shape));
    }

    /// <summary>
    /// Gets the Input value.
    /// 获取 Input 值。
    /// </summary>
    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetNetworkInput(Line, _handle, index));
    }

    /// <summary>
    /// Gets a TensorRT 11 network input tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 网络输入张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="index">The zero-based input index. 从零开始的输入索引。</param>
    /// <returns>The input tensor shape reported by TensorRT. TensorRT 报告的输入张量形状。</returns>
    public TensorRtDims64 GetInputShape64(int index)
    {
        return NativeBridgeApi.GetNetworkInputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one TensorRT 11 network input tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 网络输入张量的单个维度 extent。
    /// </summary>
    /// <param name="index">The zero-based input index. 从零开始的输入索引。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The input dimension extent reported by TensorRT. TensorRT 报告的输入维度 extent。</returns>
    public long GetInputDimensionExtent64(int index, int dimensionIndex)
    {
        return NativeBridgeApi.GetNetworkInputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets the Output value.
    /// 获取 Output 值。
    /// </summary>
    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(Line, NativeBridgeApi.GetNetworkOutput(Line, _handle, index));
    }

    /// <summary>
    /// Gets a TensorRT 11 network output tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 网络输出张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="index">The zero-based output index. 从零开始的输出索引。</param>
    /// <returns>The output tensor shape reported by TensorRT. TensorRT 报告的输出张量形状。</returns>
    public TensorRtDims64 GetOutputShape64(int index)
    {
        return NativeBridgeApi.GetNetworkOutputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one TensorRT 11 network output tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 网络输出张量的单个维度 extent。
    /// </summary>
    /// <param name="index">The zero-based output index. 从零开始的输出索引。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The output dimension extent reported by TensorRT. TensorRT 报告的输出维度 extent。</returns>
    public long GetOutputDimensionExtent64(int index, int dimensionIndex)
    {
        return NativeBridgeApi.GetNetworkOutputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets the Layer value.
    /// 获取 Layer 值。
    /// </summary>
    public TensorRtLayer GetLayer(int index)
    {
        SafeTensorRtObjectHandleLease ownerLease = SafeTensorRtObjectHandleLease.Create(_handle);
        try
        {
            SafeTensorRtObjectHandle layer = NativeBridgeApi.GetNetworkLayer(Line, _handle, index);
            return new TensorRtLayer(Line, layer, ownerLease);
        }
        catch
        {
            ownerLease.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Marks the Output value.
    /// 标记 Output 值。
    /// </summary>
    public void MarkOutput(TensorRtTensor tensor)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(nameof(tensor));
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Output tensor must belong to the same TensorRT API line as the network.");
        }

        NativeBridgeApi.MarkNetworkOutput(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Unmarks the Output value.
    /// 取消标记 Output 值。
    /// </summary>
    public void UnmarkOutput(TensorRtTensor tensor)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(nameof(tensor));
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Output tensor must belong to the same TensorRT API line as the network.");
        }

        NativeBridgeApi.UnmarkNetworkOutput(Line, _handle, tensor.Handle);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateInputTensor(TensorRtTensor tensor, string argumentName)
    {
        if (tensor == null)
        {
            throw new ArgumentNullException(argumentName);
        }

        if (tensor.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.", argumentName);
        }
    }
}
