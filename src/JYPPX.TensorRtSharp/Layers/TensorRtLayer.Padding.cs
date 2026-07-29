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
    /// Gets the Padding Pre Padding value.
    /// 获取 Padding Pre Padding 值。
    /// </summary>
    public TensorRtDims GetPaddingPrePadding()
    {
        return NativeBridgeApi.GetPaddingPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Padding Pre Padding value.
    /// 设置 Padding Pre Padding 值。
    /// </summary>
    public void SetPaddingPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Padding Post Padding value.
    /// 获取 Padding Post Padding 值。
    /// </summary>
    public TensorRtDims GetPaddingPostPadding()
    {
        return NativeBridgeApi.GetPaddingPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Padding Post Padding value.
    /// 设置 Padding Post Padding 值。
    /// </summary>
    public void SetPaddingPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetPaddingPostPadding(Line, _handle, padding);
    }

}
