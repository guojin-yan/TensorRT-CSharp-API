using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaStream
{
    /// <summary>
    /// Gets the supported CUDA stream-priority range for the current device.
    /// 获取当前设备支持的 CUDA stream 优先级范围。
    /// </summary>
    /// <returns>The stream-priority range. stream 优先级范围。</returns>
    public static CudaStreamPriorityRange GetPriorityRange()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.GetStreamPriorityRange(out int leastPriority, out int greatestPriority);
        return new CudaStreamPriorityRange(leastPriority, greatestPriority);
    }

    /// <summary>
    /// Exchanges the per-thread CUDA stream-capture mode and returns the previous mode.
    /// 交换当前线程的 CUDA stream capture mode，并返回交换前的模式。
    /// </summary>
    /// <param name="mode">The mode to install for the current thread. 要设置到当前线程的模式。</param>
    /// <returns>The previous CUDA stream-capture mode. 之前的 CUDA stream capture mode。</returns>
    public static CudaStreamCaptureMode ExchangeThreadCaptureMode(CudaStreamCaptureMode mode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.ExchangeThreadStreamCaptureMode(mode);
    }

    /// <summary>
    /// Returns whether all queued work on this stream has completed.
    /// 返回当前 stream 上排队的工作是否已全部完成。
    /// </summary>
    /// <returns><see langword="true"/> when the stream is ready. 当 stream 已就绪时返回 <see langword="true"/>。</returns>
    public bool IsReady()
    {
        return NativeCudaApi.QueryStream(_handle);
    }

}
