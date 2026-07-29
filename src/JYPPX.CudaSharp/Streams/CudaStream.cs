using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA stream handle.
/// CUDA stream 句柄的托管封装。
/// </summary>
public sealed partial class CudaStream : IDisposable
{
    private readonly SafeCudaStreamHandle _handle;
    private readonly object _captureLifecycleGate = new object();
    private int _activeCaptureToGraphSessions;
    private bool _disposed;

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

    internal CudaStream(SafeCudaStreamHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
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
    /// Releases the CUDA stream handle.
    /// 释放 CUDA stream 句柄。
    /// </summary>
    public void Dispose()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeCaptureToGraphSessions != 0)
            {
                throw new InvalidOperationException("The CUDA stream cannot be disposed while a stream-to-graph capture session is active.");
            }

            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    internal void EnterCaptureToGraphSession()
    {
        lock (_captureLifecycleGate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CudaStream));
            }

            _activeCaptureToGraphSessions++;
        }
    }

    internal void ExitCaptureToGraphSession()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeCaptureToGraphSessions > 0)
            {
                _activeCaptureToGraphSessions--;
            }
        }
    }
}
