using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA event handle.
/// </summary>
public sealed class CudaEvent : IDisposable
{
    private readonly SafeCudaEventHandle _handle;

    /// <summary>
    /// Creates a CUDA event with default flags.
    /// 使用默认标志创建 CUDA event。
    /// </summary>
    public CudaEvent()
        : this(CudaEventCreationFlags.Default)
    {
    }

    /// <summary>
    /// Creates a CUDA event with explicit creation flags.
    /// 使用显式创建标志创建 CUDA event。
    /// </summary>
    /// <param name="flags">The CUDA event creation flags. CUDA event 创建标志。</param>
    public CudaEvent(CudaEventCreationFlags flags)
    {
        NativeBridgeLoader.EnsureInitialized();
        _handle = flags == CudaEventCreationFlags.Default ? NativeCudaApi.CreateEvent() : NativeCudaApi.CreateEvent(flags);
    }

    /// <summary>
    /// Gets the effective CUDA event flags.
    /// 获取实际生效的 CUDA event 标志。
    /// </summary>
    public CudaEventCreationFlags Flags => NativeCudaApi.GetEventFlags(_handle);

    /// <summary>
    /// Records this event on a CUDA stream.
    /// 在 CUDA stream 上记录当前 event。
    /// </summary>
    /// <param name="stream">The stream that receives the record operation. 接收 record 操作的 stream。</param>
    public void Record(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.RecordEvent(_handle, stream.Handle);
    }

    /// <summary>
    /// Records this CUDA event into a stream with explicit event-record flags.
    /// 使用显式 event-record flags 将当前 CUDA event 记录到 stream。
    /// </summary>
    /// <param name="stream">The stream that receives the event record operation. 接收 event record 操作的 stream。</param>
    /// <param name="flags">CUDA event-record flags. CUDA event record 标志。</param>
    public void Record(CudaStream stream, CudaEventRecordFlags flags)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.RecordEventWithFlags(_handle, stream.Handle, (uint)flags);
    }

    internal SafeCudaEventHandle Handle => _handle;

    /// <summary>
    /// Blocks until the CUDA event is completed.
    /// 阻塞直到 CUDA event 完成。
    /// </summary>
    public void Synchronize()
    {
        NativeCudaApi.SynchronizeEvent(_handle);
    }

    /// <summary>
    /// Gets whether the CUDA event has already completed.
    /// 获取该 CUDA event 是否已经完成。
    /// </summary>
    /// <returns><see langword="true"/> when the event is ready. event 已就绪时返回 <see langword="true"/>。</returns>
    public bool IsReady()
    {
        return NativeCudaApi.QueryEvent(_handle);
    }

    /// <summary>
    /// Measures elapsed time since a start event.
    /// 计算从起始 event 到当前 event 的耗时。
    /// </summary>
    /// <param name="startEvent">The starting CUDA event. 起始 CUDA event。</param>
    /// <returns>The elapsed time in milliseconds. 耗时（毫秒）。</returns>
    public float ElapsedTimeSince(CudaEvent startEvent)
    {
        if (startEvent == null)
        {
            throw new ArgumentNullException(nameof(startEvent));
        }

        return NativeCudaApi.GetEventElapsedTime(startEvent.Handle, _handle);
    }

    /// <summary>
    /// Releases the native CUDA event handle.
    /// 释放原生 CUDA event 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
