using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private TensorRtProgressMonitor? _progressMonitorKeepAlive;
    private bool _disposed;

    internal TensorRtBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this builder configuration.
    /// 获取当前 builder 配置使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Releases the TensorRT builder-configuration handle.
    /// 释放 TensorRT builder 配置句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        TensorRtProgressMonitor? monitor = _progressMonitorKeepAlive;
        _disposed = true;
        if (monitor != null)
        {
            TryClearProgressMonitorForDispose();
        }

        _handle.Dispose();
        GC.KeepAlive(monitor);
        DetachProgressMonitor();
        GC.SuppressFinalize(this);
    }

    private void TryClearProgressMonitorForDispose()
    {
        try
        {
            NativeBridgeApi.ClearBuilderConfigProgressMonitor(Line, _handle);
        }
        catch (BridgeProbeException)
        {
            // Dispose must still release the config handle. Keep the progress monitor alive until
            // the config handle is released so TensorRT never observes a freed borrowed monitor.
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtBuilderConfig));
        }
    }

    private TensorRtProgressMonitor? DetachProgressMonitor()
    {
        TensorRtProgressMonitor? monitor = _progressMonitorKeepAlive;
        if (monitor != null)
        {
            _progressMonitorKeepAlive = null;
            monitor.DetachBorrower();
        }

        return monitor;
    }

    private void ValidateLayer(TensorRtLayer layer)
    {
        if (layer == null)
        {
            throw new ArgumentNullException(nameof(layer));
        }

        if (layer.Line != Line)
        {
            throw new ArgumentException("Layer must belong to the same TensorRT API line as the builder config.");
        }
    }
}
