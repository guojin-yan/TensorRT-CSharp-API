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
    /// Returns whether TensorRT currently has an algorithm selector attached, without exposing the borrowed selector pointer.
    /// 返回 TensorRT 当前是否附加了 algorithm selector；该属性只报告 presence，不暴露 borrowed selector 指针。
    /// </summary>
    /// <remarks>
    /// This TensorRT 8/10 compatibility probe does not transfer ownership and cannot be used to invoke selector callbacks.
    /// 这是 TensorRT 8/10 compatibility 查询，不转移生命周期，也不能用于调用 selector 回调。
    /// </remarks>
    public bool HasAlgorithmSelectorCompatibility => NativeBridgeApi.HasBuilderConfigAlgorithmSelectorCompatibility(Line, _handle);

    /// <summary>
    /// Returns whether TensorRT currently has an INT8 calibrator attached, without exposing the borrowed calibrator pointer.
    /// 返回 TensorRT 当前是否附加了 INT8 calibrator；该属性只报告 presence，不暴露 borrowed calibrator 指针。
    /// </summary>
    /// <remarks>
    /// This TensorRT 8/10 compatibility probe does not transfer ownership and cannot be used to invoke calibrator callbacks.
    /// 这是 TensorRT 8/10 compatibility 查询，不转移生命周期，也不能用于调用 calibrator 回调。
    /// </remarks>
    public bool HasInt8CalibratorCompatibility => NativeBridgeApi.HasBuilderConfigInt8CalibratorCompatibility(Line, _handle);

}
