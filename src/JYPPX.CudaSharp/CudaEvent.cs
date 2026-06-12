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

    public CudaEvent()
        : this(CudaEventCreationFlags.Default)
    {
    }

    public CudaEvent(CudaEventCreationFlags flags)
    {
        NativeBridgeLoader.EnsureInitialized();
        _handle = flags == CudaEventCreationFlags.Default ? NativeCudaApi.CreateEvent() : NativeCudaApi.CreateEvent(flags);
    }

    public CudaEventCreationFlags Flags => NativeCudaApi.GetEventFlags(_handle);

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

    public void Synchronize()
    {
        NativeCudaApi.SynchronizeEvent(_handle);
    }

    public bool IsReady()
    {
        return NativeCudaApi.QueryEvent(_handle);
    }

    public float ElapsedTimeSince(CudaEvent startEvent)
    {
        if (startEvent == null)
        {
            throw new ArgumentNullException(nameof(startEvent));
        }

        return NativeCudaApi.GetEventElapsedTime(startEvent.Handle, _handle);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
