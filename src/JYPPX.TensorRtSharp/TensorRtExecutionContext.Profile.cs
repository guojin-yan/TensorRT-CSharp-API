using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    public int OptimizationProfileIndex => NativeBridgeApi.GetExecutionContextOptimizationProfile(Line, _handle);

    public bool EnqueueEmitsProfile
    {
        get => NativeBridgeApi.GetExecutionContextEnqueueEmitsProfile(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextEnqueueEmitsProfile(Line, _handle, value);
    }

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
