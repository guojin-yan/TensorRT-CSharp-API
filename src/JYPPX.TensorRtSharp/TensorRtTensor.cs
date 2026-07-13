using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a TensorRT network tensor handle.
/// 表示一个 TensorRT 网络张量句柄。
/// </summary>
public sealed partial class TensorRtTensor : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly SafeTensorRtObjectHandleLease? _ownerLease;

    internal TensorRtTensor(
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
    /// Gets the TensorRT adapter line that owns this tensor.
    /// 获取拥有此张量的 TensorRT 适配线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether this borrowed tensor wrapper keeps its native owner alive.
    /// 获取此 borrowed tensor 包装是否会保持其 native owner 存活。
    /// </summary>
    public bool IsOwnerLifetimeBound => _ownerLease != null;

    /// <summary>
    /// Gets or sets the tensor name.
    /// 获取或设置张量名称。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetTensorName(Line, _handle);
        set => NativeBridgeApi.SetTensorName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the tensor element data type.
    /// 获取或设置张量元素数据类型。
    /// </summary>
    public TensorRtDataType DataType
    {
        get => NativeBridgeApi.GetTensorDataType(Line, _handle);
        set => NativeBridgeApi.SetTensorDataType(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the tensor shape using TensorRT's classic 32-bit dimensions.
    /// 使用 TensorRT 经典 32 位维度获取或设置张量形状。
    /// </summary>
    public TensorRtDims Shape
    {
        get => NativeBridgeApi.GetTensorShape(Line, _handle);
        set => NativeBridgeApi.SetTensorShape(Line, _handle, value);
    }

    /// <summary>
    /// Gets the TensorRT 11 tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <remarks>
    /// This API is available only for the TensorRT 11 adapter line. It keeps unknown-rank shapes and large extents intact.
    /// 该 API 仅适用于 TensorRT 11 适配线；它会保留 unknown-rank shape 和大维度 extent。
    /// </remarks>
    public TensorRtDims64 Shape64 => NativeBridgeApi.GetTensorShape64(Line, _handle);

    /// <summary>
    /// Gets one TensorRT 11 tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 张量的单个维度 extent。
    /// </summary>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The dimension extent reported by TensorRT. TensorRT 报告的维度 extent。</returns>
    public long GetDimensionExtent64(int dimensionIndex)
    {
        return NativeBridgeApi.GetTensorDimensionExtent64(Line, _handle, dimensionIndex);
    }

    /// <summary>
    /// Gets or sets whether the tensor is expected in device or host memory.
    /// 获取或设置张量应位于设备内存还是主机内存。
    /// </summary>
    public TensorRtTensorLocation Location
    {
        get => NativeBridgeApi.GetTensorLocation(Line, _handle);
        set => NativeBridgeApi.SetTensorLocation(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the allowed tensor memory formats bitmask.
    /// 获取或设置允许的张量内存格式位掩码。
    /// </summary>
    public TensorRtTensorFormats AllowedFormats
    {
        get => NativeBridgeApi.GetTensorAllowedFormats(Line, _handle);
        set => NativeBridgeApi.SetTensorAllowedFormats(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets TensorRT's legacy broadcast-across-batch flag for this tensor.
    /// 获取或设置 TensorRT 针对此张量的旧版 broadcast-across-batch 标志。
    /// </summary>
    /// <remarks>
    /// This flag is deprecated in modern explicit-batch workflows, but it is still exposed for version coverage and model compatibility.
    /// 该标志在现代 explicit-batch 流程中已不推荐使用；这里保留它是为了版本覆盖和模型兼容。
    /// </remarks>
    public bool BroadcastAcrossBatch
    {
        get => NativeBridgeApi.GetTensorBroadcastAcrossBatch(Line, _handle);
        set => NativeBridgeApi.SetTensorBroadcastAcrossBatch(Line, _handle, value);
    }

    /// <summary>
    /// Gets whether an explicit dynamic range is set on this tensor.
    /// 获取此张量是否已经设置显式 dynamic range。
    /// </summary>
    /// <remarks>
    /// TensorRT 8 and 10 expose this query. TensorRT 11 removed the corresponding public query, so this property throws <see cref="BridgeProbeException"/> for TensorRT 11.
    /// TensorRT 8 和 10 提供该查询；TensorRT 11 已移除对应公开查询，因此该属性在 TensorRT 11 下会抛出 <see cref="BridgeProbeException"/>。
    /// </remarks>
    public bool IsDynamicRangeSet => NativeBridgeApi.IsTensorDynamicRangeSet(Line, _handle);

    /// <summary>
    /// Gets the minimum dynamic range value currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的 dynamic range 最小值。
    /// </summary>
    public float DynamicRangeMinimum => NativeBridgeApi.GetTensorDynamicRangeMin(Line, _handle);

    /// <summary>
    /// Gets the maximum dynamic range value currently reported by TensorRT.
    /// 获取 TensorRT 当前报告的 dynamic range 最大值。
    /// </summary>
    public float DynamicRangeMaximum => NativeBridgeApi.GetTensorDynamicRangeMax(Line, _handle);

    /// <summary>
    /// Sets an explicit dynamic range on the tensor.
    /// 为张量设置显式 dynamic range。
    /// </summary>
    /// <param name="minimum">The lower bound. 下界。</param>
    /// <param name="maximum">The upper bound. 上界。</param>
    public void SetDynamicRange(float minimum, float maximum)
    {
        NativeBridgeApi.SetTensorDynamicRange(Line, _handle, minimum, maximum);
    }

    /// <summary>
    /// Clears the explicit dynamic range on the tensor.
    /// 清除张量上的显式 dynamic range。
    /// </summary>
    public void ResetDynamicRange()
    {
        NativeBridgeApi.ResetTensorDynamicRange(Line, _handle);
    }

    /// <summary>
    /// Releases the managed wrapper for the TensorRT tensor handle.
    /// 释放 TensorRT 张量句柄的托管包装。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        _ownerLease?.Dispose();
        GC.SuppressFinalize(this);
    }
}
