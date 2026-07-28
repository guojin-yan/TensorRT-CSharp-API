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
    /// Gets the Lrn Window Size value.
    /// 获取 Lrn Window Size 值。
    /// </summary>
    public int GetLrnWindowSize()
    {
        return NativeBridgeApi.GetLrnWindowSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Lrn Window Size value.
    /// 设置 Lrn Window Size 值。
    /// </summary>
    public void SetLrnWindowSize(int windowSize)
    {
        if (windowSize < 1 || windowSize > 15 || windowSize % 2 == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize), "LRN window size must be an odd value in [1, 15].");
        }

        NativeBridgeApi.SetLrnWindowSize(Line, _handle, windowSize);
    }

    /// <summary>
    /// Gets the Lrn Alpha value.
    /// 获取 Lrn Alpha 值。
    /// </summary>
    public float GetLrnAlpha()
    {
        return NativeBridgeApi.GetLrnAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the Lrn Alpha value.
    /// 设置 Lrn Alpha 值。
    /// </summary>
    public void SetLrnAlpha(float alpha)
    {
        NativeBridgeApi.SetLrnAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the Lrn Beta value.
    /// 获取 Lrn Beta 值。
    /// </summary>
    public float GetLrnBeta()
    {
        return NativeBridgeApi.GetLrnBeta(Line, _handle);
    }

    /// <summary>
    /// Sets the Lrn Beta value.
    /// 设置 Lrn Beta 值。
    /// </summary>
    public void SetLrnBeta(float beta)
    {
        NativeBridgeApi.SetLrnBeta(Line, _handle, beta);
    }

    /// <summary>
    /// Gets the Lrn K value.
    /// 获取 Lrn K 值。
    /// </summary>
    public float GetLrnK()
    {
        return NativeBridgeApi.GetLrnK(Line, _handle);
    }

    /// <summary>
    /// Sets the Lrn K value.
    /// 设置 Lrn K 值。
    /// </summary>
    public void SetLrnK(float k)
    {
        NativeBridgeApi.SetLrnK(Line, _handle, k);
    }
}
