using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets or sets the Optimization Profile Index value.
    /// 获取或设置 Optimization Profile Index 值。
    /// </summary>
    public int OptimizationProfileIndex => NativeBridgeApi.GetExecutionContextOptimizationProfile(Line, _handle);

    /// <summary>
    /// Gets or sets the Enqueue Emits Profile value.
    /// 获取或设置 Enqueue Emits Profile 值。
    /// </summary>
    public bool EnqueueEmitsProfile
    {
        get => NativeBridgeApi.GetExecutionContextEnqueueEmitsProfile(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextEnqueueEmitsProfile(Line, _handle, value);
    }

    /// <summary>
    /// Sets the Optimization Profile Async value.
    /// 设置 Optimization Profile Async 值。
    /// </summary>
    public void SetOptimizationProfileAsync(int profileIndex, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.SetExecutionContextOptimizationProfileAsync(Line, _handle, profileIndex, stream.Handle);
    }

    /// <summary>
    /// Reports accumulated execution data to the profiler attached to the context, when one exists.
    /// 将当前执行上下文已累计的执行数据上报给关联的 profiler（如果存在）。
    /// </summary>
    /// <returns>
    /// <see langword="true"/> when TensorRT reports that profiler data was emitted; otherwise <see langword="false"/>.
    /// 当 TensorRT 确认已上报 profiler 数据时返回 <see langword="true"/>，否则返回 <see langword="false"/>。
    /// </returns>
    public bool ReportToProfiler()
    {
        return NativeBridgeApi.ReportExecutionContextToProfiler(Line, _handle);
    }
}
