using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private TensorRtProfiler? _profilerKeepAlive;
    private readonly object _auxiliaryStreamLeaseLock = new object();
    private TensorRtAuxiliaryStreamHandleLease? _auxiliaryStreamLease;
    private int _auxiliaryStreamAssignedCount;
    private bool _auxiliaryStreamsCleared = true;
    private bool _auxiliaryStreamContextDisposed;
    private string _auxiliaryStreamDiagnostic = "No caller-provided auxiliary CUDA streams are assigned.";

    internal TensorRtExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    /// <summary>
    /// Gets or sets the Line value.
    /// 获取或设置 Line 值。
    /// </summary>
    public TensorRtApiLine Line { get; }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets or sets the Name value.
    /// 获取或设置 Name 值。
    /// </summary>
    public string Name
    {
        get => NativeBridgeApi.GetExecutionContextName(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the Debug Sync value.
    /// 获取或设置 Debug Sync 值。
    /// </summary>
    public bool DebugSync
    {
        get => NativeBridgeApi.GetExecutionContextDebugSync(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextDebugSync(Line, _handle, value);
    }

    /// <summary>
    /// Gets the All Input Dimensions Specified value.
    /// 获取 All Input Dimensions Specified 值。
    /// </summary>
    public bool AllInputDimensionsSpecified => NativeBridgeApi.AllInputDimensionsSpecified(Line, _handle);

    /// <summary>
    /// Gets or sets the All Input Shapes Specified value.
    /// 获取或设置 All Input Shapes Specified 值。
    /// </summary>
    public bool AllInputShapesSpecified => NativeBridgeApi.AllInputShapesSpecified(Line, _handle);

    /// <summary>
    /// Gets or sets the Device Memory Size In Bytes value.
    /// 获取或设置 Device Memory Size In Bytes 值。
    /// </summary>
    public ulong DeviceMemorySizeInBytes => NativeBridgeApi.GetExecutionContextDeviceMemorySize(Line, _handle);

    /// <summary>
    /// Gets or sets the Persistent Cache Limit In Bytes value.
    /// 获取或设置 Persistent Cache Limit In Bytes 值。
    /// </summary>
    public ulong PersistentCacheLimitInBytes
    {
        get => NativeBridgeApi.GetExecutionContextPersistentCacheLimit(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextPersistentCacheLimit(Line, _handle, value);
    }

    /// <summary>
    /// Releases the native TensorRT resources held by this object.
    /// 释放此对象持有的 native TensorRT 资源。
    /// </summary>
    public void Dispose()
    {
        TensorRtAuxiliaryStreamHandleLease? auxiliaryStreamLease;
        lock (_auxiliaryStreamLeaseLock)
        {
            if (_auxiliaryStreamContextDisposed)
            {
                return;
            }

            _auxiliaryStreamContextDisposed = true;
            auxiliaryStreamLease = _auxiliaryStreamLease;
            TryClearAuxStreamsForDispose();
            _auxiliaryStreamLease = null;
            _auxiliaryStreamAssignedCount = 0;
            _auxiliaryStreamsCleared = true;
            _auxiliaryStreamDiagnostic = "Execution context disposed; auxiliary stream leases were released after native context teardown.";
        }

        TensorRtProfiler? profiler = _profilerKeepAlive;
        try
        {
            if (profiler != null)
            {
                TryClearProfilerForDispose();
            }
        }
        finally
        {
            try
            {
                _handle.Dispose();
                GC.KeepAlive(profiler);
            }
            finally
            {
                auxiliaryStreamLease?.Dispose();
                DetachProfiler();
                GC.SuppressFinalize(this);
            }
        }
    }

    private void TryClearAuxStreamsForDispose()
    {
        try
        {
            NativeBridgeApi.ClearExecutionContextAuxStreams(Line, _handle);
        }
        catch (Exception exception) when (
            exception is BridgeProbeException ||
            exception is TensorRtException ||
            exception is EntryPointNotFoundException ||
            exception is DllNotFoundException ||
            exception is BadImageFormatException ||
            exception is ObjectDisposedException)
        {
            // Keep any stream lease until after context teardown even when native clear is unavailable.
        }
    }

    private void TryClearProfilerForDispose()
    {
        try
        {
            NativeBridgeApi.ClearExecutionContextProfiler(Line, _handle);
        }
        catch (BridgeProbeException)
        {
            // Dispose must still release the context handle. Keep the profiler alive until after
            // the context handle is released so TensorRT never observes a freed borrowed profiler.
        }
    }

    private TensorRtProfiler? DetachProfiler()
    {
        TensorRtProfiler? profiler = _profilerKeepAlive;
        if (profiler != null)
        {
            _profilerKeepAlive = null;
            profiler.DetachBorrower();
        }

        return profiler;
    }
}
