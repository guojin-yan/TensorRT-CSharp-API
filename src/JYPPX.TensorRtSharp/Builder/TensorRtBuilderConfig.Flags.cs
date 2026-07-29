using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Enables or disables one TensorRT builder flag.
    /// 启用或禁用一个 TensorRT builder 标志。
    /// </summary>
    /// <param name="flag">The builder flag to change. 要修改的 builder 标志。</param>
    /// <param name="enabled">Whether the flag should be enabled. 是否启用该标志。</param>
    public void SetFlag(TensorRtBuilderFlag flag, bool enabled = true)
    {
        NativeBridgeApi.SetBuilderConfigFlag(Line, _handle, flag, enabled);
    }

    /// <summary>
    /// Clears one TensorRT builder flag.
    /// 清除一个 TensorRT builder 标志。
    /// </summary>
    /// <param name="flag">The builder flag to clear. 要清除的 builder 标志。</param>
    public void ClearFlag(TensorRtBuilderFlag flag)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            NativeBridgeApi.ClearBuilderConfigFlag(Line, _handle, flag);
            return;
        }

        SetFlag(flag, false);
    }

    /// <summary>
    /// Returns whether one TensorRT builder flag is currently enabled.
    /// 返回某个 TensorRT builder 标志当前是否启用。
    /// </summary>
    /// <param name="flag">The builder flag to query. 要查询的 builder 标志。</param>
    /// <returns><see langword="true"/> when the flag is enabled. 当该标志已启用时返回 <see langword="true"/>。</returns>
    public bool GetFlag(TensorRtBuilderFlag flag)
    {
        return NativeBridgeApi.GetBuilderConfigFlag(Line, _handle, flag);
    }

}
