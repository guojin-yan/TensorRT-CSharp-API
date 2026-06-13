using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA stream handle.
/// CUDA stream 句柄的托管封装。
/// </summary>
public sealed class CudaStream : IDisposable
{
    private readonly SafeCudaStreamHandle _handle;

    /// <summary>
    /// Creates a CUDA stream with default flags.
    /// 使用默认标志创建一个 CUDA stream。
    /// </summary>
    public CudaStream()
        : this(CudaStreamCreationFlags.Default)
    {
    }

    /// <summary>
    /// Creates a CUDA stream with explicit creation flags.
    /// 使用显式创建标志创建一个 CUDA stream。
    /// </summary>
    /// <param name="flags">The stream-creation flags. stream 创建标志。</param>
    public CudaStream(CudaStreamCreationFlags flags)
    {
        NativeBridgeLoader.EnsureInitialized();
        _handle = flags == CudaStreamCreationFlags.Default ? NativeCudaApi.CreateStream() : NativeCudaApi.CreateStream(flags);
    }

    /// <summary>
    /// Creates a CUDA stream with explicit flags and priority.
    /// 使用显式标志和优先级创建一个 CUDA stream。
    /// </summary>
    /// <param name="flags">The stream-creation flags. stream 创建标志。</param>
    /// <param name="priority">The CUDA stream priority. CUDA stream 优先级。</param>
    public CudaStream(CudaStreamCreationFlags flags, int priority)
    {
        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.CreateStreamWithPriority((uint)flags, priority);
    }

    internal SafeCudaStreamHandle Handle => _handle;

    /// <summary>
    /// Gets the CUDA flags associated with this stream.
    /// 获取当前 stream 关联的 CUDA 标志。
    /// </summary>
    public CudaStreamCreationFlags Flags => NativeCudaApi.GetStreamFlags(_handle);

    /// <summary>
    /// Gets the CUDA priority associated with this stream.
    /// 获取当前 stream 关联的 CUDA 优先级。
    /// </summary>
    public int Priority => NativeCudaApi.GetStreamPriority(_handle);

    /// <summary>
    /// Gets the CUDA runtime stream id when the bridge was built with CUDA 12.0 or later.
    /// 当桥接库使用 CUDA 12.0 或更高版本构建时，获取 CUDA runtime stream id。
    /// </summary>
    public ulong Id => NativeCudaApi.GetStreamId(_handle);

    /// <summary>
    /// Gets the device ordinal associated with this stream when supported by the CUDA runtime.
    /// 在 CUDA runtime 支持时，获取该 stream 关联的设备序号。
    /// </summary>
    public int DeviceOrdinal => NativeCudaApi.GetStreamDevice(_handle);

    /// <summary>
    /// Gets the current CUDA graph-capture status for this stream.
    /// 获取当前 stream 的 CUDA graph 捕获状态。
    /// </summary>
    public CudaStreamCaptureStatus CaptureStatus => NativeCudaApi.GetStreamCaptureStatus(_handle);

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

    /// <summary>
    /// Gets CUDA graph-capture metadata for this stream.
    /// 获取当前 stream 的 CUDA graph 捕获元数据。
    /// </summary>
    /// <returns>The current capture status and capture id. 当前捕获状态和捕获 ID。</returns>
    public CudaStreamCaptureInfo GetCaptureInfo()
    {
        return NativeCudaApi.GetStreamCaptureInfo(_handle);
    }

    /// <summary>
    /// Tries to read CUDA graph-capture metadata without throwing when the runtime or symbol is unavailable.
    /// 在 CUDA runtime 或相关符号不可用时以诊断字符串返回失败，而不是抛出异常。
    /// </summary>
    /// <param name="captureInfo">The capture metadata when the query succeeds. 查询成功时返回的 capture 元数据。</param>
    /// <param name="diagnostic">An empty string on success, or the CUDA bridge diagnostic on failure. 成功时为空字符串，失败时为 CUDA 桥接诊断。</param>
    /// <returns><c>true</c> when capture metadata was read successfully. 成功读取 capture 元数据时返回 <c>true</c>。</returns>
    public bool TryGetCaptureInfo(out CudaStreamCaptureInfo captureInfo, out string diagnostic)
    {
        try
        {
            captureInfo = GetCaptureInfo();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            captureInfo = default;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Inserts a wait on a CUDA event into this stream.
    /// 在当前 stream 中插入对 CUDA event 的等待。
    /// </summary>
    /// <param name="cudaEvent">The event to wait on. 要等待的 event。</param>
    public void WaitFor(CudaEvent cudaEvent)
    {
        if (cudaEvent == null)
        {
            throw new ArgumentNullException(nameof(cudaEvent));
        }

        NativeCudaApi.WaitStreamForEvent(_handle, cudaEvent.Handle);
    }

    /// <summary>
    /// Copies stream attributes from another CUDA stream into this stream.
    /// 将另一个 CUDA stream 的属性复制到当前 stream。
    /// </summary>
    /// <param name="source">The source stream whose attributes should be copied. 要复制属性的源 stream。</param>
    public void CopyAttributesFrom(CudaStream source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        NativeCudaApi.CopyStreamAttributes(_handle, source.Handle);
    }

    /// <summary>
    /// Blocks the calling thread until this stream completes.
    /// 阻塞调用线程，直到当前 stream 完成。
    /// </summary>
    public void Synchronize()
    {
        NativeCudaApi.SynchronizeStream(_handle);
    }

    /// <summary>
    /// Measures elapsed GPU time for work submitted to this stream.
    /// 测量提交到当前 stream 的工作所消耗的 GPU 时间。
    /// </summary>
    /// <param name="submitWork">The delegate that submits CUDA work to this stream. 向当前 stream 提交 CUDA 工作的委托。</param>
    /// <returns>The elapsed GPU time in milliseconds. GPU 经过时间，单位为毫秒。</returns>
    public float MeasureElapsedTime(Action<CudaStream> submitWork)
    {
        if (submitWork == null)
        {
            throw new ArgumentNullException(nameof(submitWork));
        }

        using CudaEvent start = new CudaEvent();
        using CudaEvent stop = new CudaEvent();
        start.Record(this);
        submitWork(this);
        stop.Record(this);
        stop.Synchronize();
        return stop.ElapsedTimeSince(start);
    }

    /// <summary>
    /// Begins CUDA graph capture on this stream.
    /// 在当前 stream 上开始 CUDA graph capture。
    /// </summary>
    /// <param name="mode">The capture-validation mode. capture 校验模式。</param>
    public void BeginCapture(CudaStreamCaptureMode mode = CudaStreamCaptureMode.Global)
    {
        NativeCudaApi.BeginStreamCapture(_handle, mode);
    }

    /// <summary>
    /// Ends CUDA graph capture and returns the captured graph.
    /// 结束 CUDA graph capture 并返回捕获得到的 graph。
    /// </summary>
    /// <returns>The captured CUDA graph. 捕获得到的 CUDA graph。</returns>
    public CudaGraph EndCapture()
    {
        return new CudaGraph(NativeCudaApi.EndStreamCapture(_handle));
    }

    /// <summary>
    /// Releases the CUDA stream handle.
    /// 释放 CUDA stream 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
