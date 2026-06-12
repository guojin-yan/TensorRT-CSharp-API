using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    public int GetQuantizeAxis()
    {
        return NativeBridgeApi.GetQuantizeAxis(Line, _handle);
    }

    public void SetQuantizeAxis(int axis)
    {
        ValidateQuantizationAxis(axis, nameof(axis));
        NativeBridgeApi.SetQuantizeAxis(Line, _handle, axis);
    }

    public int GetDequantizeAxis()
    {
        return NativeBridgeApi.GetDequantizeAxis(Line, _handle);
    }

    public void SetDequantizeAxis(int axis)
    {
        ValidateQuantizationAxis(axis, nameof(axis));
        NativeBridgeApi.SetDequantizeAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the TensorRT 11 quantization block shape configured on a dequantize layer.
    /// 获取 TensorRT 11 dequantize 层配置的量化 block shape。
    /// </summary>
    /// <returns>The current block shape, or an empty dimensions value when TensorRT uses the default.</returns>
    public TensorRtDims GetDequantizeBlockShape()
    {
        return NativeBridgeApi.GetDequantizeBlockShape(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 11 quantization block shape on a dequantize layer.
    /// 设置 TensorRT 11 dequantize 层的量化 block shape。
    /// </summary>
    /// <param name="blockShape">The block shape dimensions to request.</param>
    /// <returns><c>true</c> when TensorRT accepts the block shape; otherwise <c>false</c>.</returns>
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
