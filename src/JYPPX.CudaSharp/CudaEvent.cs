using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA event handle.
/// CUDA event 句柄的托管封装。
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
        IsIpcImported = false;
    }

    private CudaEvent(SafeCudaEventHandle handle, bool isIpcImported)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        IsIpcImported = isIpcImported;
    }

    /// <summary>
    /// Gets the effective CUDA event flags.
    /// 获取实际生效的 CUDA event 标志。
    /// </summary>
    public CudaEventCreationFlags Flags => NativeCudaApi.GetEventFlags(_handle);

    /// <summary>Gets whether this wrapper owns an imported process-local CUDA event. 获取此 wrapper 是否拥有导入的进程内 CUDA event。</summary>
    public bool IsIpcImported { get; }

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
    /// Copies an opaque CUDA IPC export token for this event.
    /// 复制当前 event 的 opaque CUDA IPC 导出 token。
    /// </summary>
    /// <remarks>
    /// The event must be created with <see cref="CudaEventCreationFlags.Interprocess"/> and
    /// <see cref="CudaEventCreationFlags.DisableTiming"/>. Keep this event alive while another process uses the token.
    /// event 必须使用 Interprocess 与 DisableTiming 标志创建；其他进程使用 token 期间必须保持当前 event 存活。
    /// </remarks>
    public CudaIpcExportToken ExportIpcToken()
    {
        return NativeCudaApi.ExportEventIpcToken(_handle);
    }

    /// <summary>Opens a process-local CUDA event from an export token. 从 export token 打开进程内 CUDA event。</summary>
    /// <param name="token">The event token transported from another process. 从其他进程传输的 event token。</param>
    /// <returns>An owner wrapper released with <c>cudaEventDestroy</c>. 使用 <c>cudaEventDestroy</c> 释放的 owner wrapper。</returns>
    public static CudaEvent ImportIpcToken(CudaIpcExportToken token)
    {
        if (token == null)
        {
            throw new ArgumentNullException(nameof(token));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaEvent(NativeCudaApi.ImportEventIpcToken(token), isIpcImported: true);
    }

    /// <summary>Tries to import a CUDA IPC event token and returns a diagnostic on CUDA failure. 尝试导入 CUDA IPC event token，并在 CUDA 失败时返回诊断。</summary>
    public static bool TryImportIpcToken(
        CudaIpcExportToken token,
        out CudaEvent? cudaEvent,
        out string diagnostic)
    {
        try
        {
            cudaEvent = ImportIpcToken(token);
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            cudaEvent = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>Tries to copy an IPC export token and returns a diagnostic on failure. 尝试复制 IPC 导出 token，失败时返回诊断。</summary>
    public bool TryExportIpcToken(out CudaIpcExportToken? token, out string diagnostic)
    {
        try
        {
            token = ExportIpcToken();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            token = null;
            diagnostic = exception.Message;
            return false;
        }
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
