using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the Quantize Axis value.
    /// 获取 Quantize Axis 值。
    /// </summary>
    public int GetQuantizeAxis()
    {
        return NativeBridgeApi.GetQuantizeAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Quantize Axis value.
    /// 设置 Quantize Axis 值。
    /// </summary>
    public void SetQuantizeAxis(int axis)
    {
        ValidateQuantizationAxis(axis, nameof(axis));
        NativeBridgeApi.SetQuantizeAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the Dequantize Axis value.
    /// 获取 Dequantize Axis 值。
    /// </summary>
    public int GetDequantizeAxis()
    {
        return NativeBridgeApi.GetDequantizeAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Dequantize Axis value.
    /// 设置 Dequantize Axis 值。
    /// </summary>
    public void SetDequantizeAxis(int axis)
    {
        ValidateQuantizationAxis(axis, nameof(axis));
        NativeBridgeApi.SetDequantizeAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the TensorRT 11 quantization block shape configured on a dequantize layer.
    /// 获取 TensorRT 11 dequantize 层配置的量化 block shape。
    /// </summary>
    /// <returns>The current block shape, or an empty dimensions value when TensorRT uses the default. 当前 block shape；当 TensorRT 使用默认值时返回空维度值。</returns>
    public TensorRtDims GetDequantizeBlockShape()
    {
        return NativeBridgeApi.GetDequantizeBlockShape(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 11 quantization block shape on a dequantize layer.
    /// 设置 TensorRT 11 dequantize 层的量化 block shape。
    /// </summary>
    /// <param name="blockShape">The block shape dimensions to request. 请求设置的 block shape 维度。</param>
    /// <returns><c>true</c> when TensorRT accepts the block shape; otherwise <c>false</c>. TensorRT 接受该 block shape 时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    public bool SetDequantizeBlockShape(TensorRtDims blockShape)
    {
        ValidateDims(blockShape, nameof(blockShape));
        return NativeBridgeApi.SetDequantizeBlockShape(Line, _handle, blockShape);
    }

    private static void ValidateQuantizationAxis(int axis, string argumentName)
    {
        if (axis < -1)
        {
            throw new ArgumentOutOfRangeException(argumentName, "Quantization axis must be -1 for per-tensor quantization or a non-negative tensor dimension index.");
        }
    }
}
