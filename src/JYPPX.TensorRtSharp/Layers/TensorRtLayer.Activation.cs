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
    /// Gets the Activation Type value.
    /// 获取 Activation Type 值。
    /// </summary>
    public TensorRtActivationType GetActivationType()
    {
        return NativeBridgeApi.GetActivationType(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Type value.
    /// 设置 Activation Type 值。
    /// </summary>
    public void SetActivationType(TensorRtActivationType activationType)
    {
        NativeBridgeApi.SetActivationType(Line, _handle, activationType);
    }

    /// <summary>
    /// Gets the Activation Alpha value.
    /// 获取 Activation Alpha 值。
    /// </summary>
    public double GetActivationAlpha()
    {
        return NativeBridgeApi.GetActivationAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Alpha value.
    /// 设置 Activation Alpha 值。
    /// </summary>
    public void SetActivationAlpha(double alpha)
    {
        NativeBridgeApi.SetActivationAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the Activation Beta value.
    /// 获取 Activation Beta 值。
    /// </summary>
    public double GetActivationBeta()
    {
        return NativeBridgeApi.GetActivationBeta(Line, _handle);
    }

    /// <summary>
    /// Sets the Activation Beta value.
    /// 设置 Activation Beta 值。
    /// </summary>
    public void SetActivationBeta(double beta)
    {
        NativeBridgeApi.SetActivationBeta(Line, _handle, beta);
    }

}
