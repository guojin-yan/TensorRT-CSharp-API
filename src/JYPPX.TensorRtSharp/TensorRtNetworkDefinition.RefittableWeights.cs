using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Marks a named weights object as refittable when TensorRT individual refit is enabled.
    /// 在启用 TensorRT 单独 refit 能力时，把指定名称的权重标记为可 refit。
    /// </summary>
    /// <param name="weightsName">TensorRT weights name. / TensorRT 权重名称。</param>
    /// <returns>True if TensorRT accepted the mark operation; otherwise false. / 如果 TensorRT 接受标记操作则返回 true，否则返回 false。</returns>
    public bool MarkWeightsRefittable(string weightsName)
    {
        return Internal.Interop.NativeBridgeApi.MarkWeightsRefittable(Line, _handle, weightsName);
    }

    /// <summary>
    /// Removes the refittable mark from a named weights object.
    /// 从指定名称的权重上移除可 refit 标记。
    /// </summary>
    /// <param name="weightsName">TensorRT weights name. / TensorRT 权重名称。</param>
    /// <returns>True if TensorRT accepted the unmark operation; otherwise false. / 如果 TensorRT 接受取消标记操作则返回 true，否则返回 false。</returns>
    public bool UnmarkWeightsRefittable(string weightsName)
    {
        return Internal.Interop.NativeBridgeApi.UnmarkWeightsRefittable(Line, _handle, weightsName);
    }

    /// <summary>
    /// Queries whether a named weights object is marked as refittable.
    /// 查询指定名称的权重是否已经标记为可 refit。
    /// </summary>
    /// <param name="weightsName">TensorRT weights name. / TensorRT 权重名称。</param>
    /// <returns>True when the named weights are currently marked as refittable. / 当指定权重当前已标记为可 refit 时返回 true。</returns>
    public bool AreWeightsMarkedRefittable(string weightsName)
    {
        return Internal.Interop.NativeBridgeApi.AreWeightsMarkedRefittable(Line, _handle, weightsName);
    }

    /// <summary>
    /// Assigns a TensorRT name to a weights object before engine build.
    /// 在 engine 构建前为 weights 对象分配 TensorRT 名称。
    /// </summary>
    /// <param name="constantLayer">A constant layer created by this network. 由当前 network 创建的 constant layer。</param>
    /// <param name="weightsName">TensorRT weights name. TensorRT 权重名称。</param>
    /// <returns>True if TensorRT accepted the name assignment; otherwise false. 如果 TensorRT 接受名称分配则返回 true，否则返回 false。</returns>
    public bool SetWeightsName(TensorRtLayer constantLayer, string weightsName)
    {
        if (constantLayer == null)
        {
            throw new ArgumentNullException(nameof(constantLayer));
        }

        return Internal.Interop.NativeBridgeApi.SetWeightsName(Line, _handle, constantLayer.Handle, weightsName);
    }
}
