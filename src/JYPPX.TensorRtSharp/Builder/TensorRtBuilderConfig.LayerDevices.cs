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
    /// Pins one layer to a specific TensorRT device type.
    /// 将一个 layer 固定到指定的 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <param name="deviceType">The TensorRT device type. TensorRT 设备类型。</param>
    public void SetLayerDeviceType(TensorRtLayer layer, TensorRtDeviceType deviceType)
    {
        ValidateLayer(layer);
        NativeBridgeApi.SetLayerDeviceType(Line, _handle, layer.Handle, deviceType);
    }

    /// <summary>
    /// Gets the TensorRT device type assigned to one layer.
    /// 获取一个 layer 当前分配到的 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <returns>The configured TensorRT device type. 已配置的 TensorRT device type。</returns>
    public TensorRtDeviceType GetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.GetLayerDeviceType(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Returns whether one layer has an explicit TensorRT device type.
    /// 返回一个 layer 是否具有显式 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <returns><see langword="true"/> when the layer has an explicit assignment. 当该 layer 具有显式分配时返回 <see langword="true"/>。</returns>
    public bool IsLayerDeviceTypeSet(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.IsLayerDeviceTypeSet(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Removes the explicit TensorRT device-type assignment for one layer.
    /// 移除一个 layer 的显式 TensorRT device type 分配。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    public void ResetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        NativeBridgeApi.ResetLayerDeviceType(Line, _handle, layer.Handle);
    }

}
