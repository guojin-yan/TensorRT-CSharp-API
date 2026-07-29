using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly SafeTensorRtObjectHandleLease? _ownerLease;

    internal TensorRtLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle handle,
        SafeTensorRtObjectHandleLease? ownerLease = null)
    {
        Line = line;
        _handle = handle;
        _ownerLease = ownerLease;
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
        get => NativeBridgeApi.GetLayerName(Line, _handle);
        set => NativeBridgeApi.SetLayerName(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Type value.
    /// 获取 Type 值。
    /// </summary>
    public TensorRtLayerType Type => NativeBridgeApi.GetLayerType(Line, _handle);

    /// <summary>
    /// Gets or sets the Input Count value.
    /// 获取或设置 Input Count 值。
    /// </summary>
    public int InputCount => NativeBridgeApi.GetLayerInputCount(Line, _handle);

    /// <summary>
    /// Gets or sets the Output Count value.
    /// 获取或设置 Output Count 值。
    /// </summary>
    public int OutputCount => NativeBridgeApi.GetLayerOutputCount(Line, _handle);

    /// <summary>
    /// Gets or sets the Precision value.
    /// 获取或设置 Precision 值。
    /// </summary>
    public TensorRtDataType Precision
    {
        get => NativeBridgeApi.GetLayerPrecision(Line, _handle);
        set => NativeBridgeApi.SetLayerPrecision(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Is Precision Set value.
    /// 获取 Is Precision Set 值。
    /// </summary>
    public bool IsPrecisionSet => NativeBridgeApi.IsLayerPrecisionSet(Line, _handle);

    /// <summary>
    /// Resets the Precision setting.
    /// 重置 Precision 设置。
    /// </summary>
    public void ResetPrecision()
    {
        NativeBridgeApi.ResetLayerPrecision(Line, _handle);
    }

    /// <summary>
    /// Gets the Input value.
    /// 获取 Input 值。
    /// </summary>
    public TensorRtTensor GetInput(int index)
    {
        return new TensorRtTensor(
            Line,
            NativeBridgeApi.GetLayerInput(Line, _handle, index),
            _ownerLease?.Clone());
    }

    /// <summary>
    /// Gets the Output value.
    /// 获取 Output 值。
    /// </summary>
    public TensorRtTensor GetOutput(int index)
    {
        return new TensorRtTensor(
            Line,
            NativeBridgeApi.GetLayerOutput(Line, _handle, index),
            _ownerLease?.Clone());
    }

    /// <summary>
    /// Sets the Output Type value.
    /// 设置 Output Type 值。
    /// </summary>
    public void SetOutputType(int index, TensorRtDataType dataType)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.SetLayerOutputType(Line, _handle, index, dataType);
    }

    /// <summary>
    /// Gets the Output Type value.
    /// 获取 Output Type 值。
    /// </summary>
    public TensorRtDataType GetOutputType(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputType(Line, _handle, index);
    }

    /// <summary>
    /// Checks whether Output Type Set is true.
    /// 检查 Output Type Set 是否为 true。
    /// </summary>
    public bool IsOutputTypeSet(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.IsLayerOutputTypeSet(Line, _handle, index);
    }

    /// <summary>
    /// Resets the Output Type setting.
    /// 重置 Output Type 设置。
    /// </summary>
    public void ResetOutputType(int index)
    {
        ValidateOutputIndex(index);
        NativeBridgeApi.ResetLayerOutputType(Line, _handle, index);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        _ownerLease?.Dispose();
        GC.SuppressFinalize(this);
    }

    internal SafeTensorRtObjectHandleLease CloneRequiredOwnerLease()
    {
        if (_ownerLease == null)
        {
            throw new InvalidOperationException(
                "This layer is not bound to a network owner. Retrieve the TensorRT 8 RNNv2 layer through TensorRtNetworkDefinition.GetLayer before querying borrowed state tensors.");
        }

        return _ownerLease.Clone();
    }

    private void ValidateOutputIndex(int index)
    {
        if (index < 0 || index >= OutputCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
    }

    private static void ValidateDims(TensorRtDims dims, string argumentName)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(argumentName);
        }
    }
}
