using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Attaches an owner-safe managed debug listener to this TensorRT 10/11 execution context.
    /// 将 owner-safe 托管 debug listener 绑定到当前 TensorRT 10/11 execution context。
    /// </summary>
    /// <param name="listener">The callback owner borrowed by TensorRT until clear or context disposal. TensorRT 借用到清理或 context dispose 为止的 callback owner。</param>
    public void SetDebugListener(TensorRtDebugListenerCallbackOwner listener)
    {
        if (listener == null)
        {
            throw new ArgumentNullException(nameof(listener));
        }

        if (TensorRtDebugListenerCallbackOwner.IsExecutingRuntimeCallbackOnCurrentThread)
        {
            throw new InvalidOperationException("A debug listener cannot be replaced from inside its own callback.");
        }

        lock (_debugListenerLeaseLock)
        {
            if (_debugListenerContextDisposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
            }

            if (ReferenceEquals(_debugListenerKeepAlive, listener))
            {
                return;
            }

            TensorRtDebugListenerCallbackOwner? previous = _debugListenerKeepAlive;
            if (previous != null)
            {
                NativeBridgeApi.DetachDebugListenerOwner(Line, previous.NativeHandle);
                _debugListenerKeepAlive = null;
                previous.DetachBorrower();
            }

            listener.ThrowIfDisposed();
            listener.AttachBorrower(Line);
            try
            {
                if (!NativeBridgeApi.AttachDebugListenerOwner(Line, listener.NativeHandle, _handle))
                {
                    throw new InvalidOperationException("TensorRT did not accept the debug listener callback owner.");
                }

                _debugListenerKeepAlive = listener;
            }
            catch
            {
                listener.DetachBorrower();
                throw;
            }
        }
    }

    /// <summary>Gets whether this wrapper owns a managed debug-listener borrow. 获取当前 wrapper 是否持有 managed debug-listener 借用。</summary>
    public bool HasManagedDebugListener
    {
        get
        {
            lock (_debugListenerLeaseLock)
            {
                return _debugListenerKeepAlive != null;
            }
        }
    }

    /// <summary>
    /// Gets whether the active TensorRT line supports per-tensor debug state on execution contexts.
    /// 获取当前 TensorRT 版本线是否支持 execution context 上的逐 tensor debug state。
    /// </summary>
    public bool SupportsTensorDebugState => Line == TensorRtApiLine.TensorRt8 || Line == TensorRtApiLine.TensorRt10 || Line == TensorRtApiLine.TensorRt11;

    /// <summary>
    /// Queries TensorRT's maximum output buffer size estimate for a named output tensor in the current context.
    /// 查询当前 execution context 中指定输出 tensor 的最大输出缓冲区大小估计值。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. / TensorRT tensor 名称。</param>
    /// <returns>
    /// The maximum size in bytes, or a negative value when TensorRT cannot determine the value for the current shape state.
    /// 最大字节数；当 TensorRT 在当前 shape 状态下无法确定该值时，可能返回负值。
    /// </returns>
    public long GetMaxOutputSize(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextMaxOutputSize(Line, _handle, tensorName);
    }

    /// <summary>
    /// Enables or disables TensorRT debug state for a named tensor on supported execution contexts.
    /// 在受支持的 execution context 上启用或禁用指定 tensor 的 debug state。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. / TensorRT tensor 名称。</param>
    /// <param name="enabled">Whether debug state should be enabled. / 是否启用 debug state。</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when the active TensorRT line does not expose tensor debug-state APIs.
    /// 当前 TensorRT 版本线未暴露 tensor debug-state API 时抛出。
    /// </exception>
    public void SetTensorDebugState(string tensorName, bool enabled)
    {
        ThrowIfTensorDebugStateUnsupported();
        NativeBridgeApi.SetExecutionContextTensorDebugState(Line, _handle, tensorName, enabled);
    }

    /// <summary>
    /// Gets the TensorRT debug state for a named tensor on supported execution contexts.
    /// 获取受支持 execution context 上指定 tensor 的 debug state。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. / TensorRT tensor 名称。</param>
    /// <returns><see langword="true"/> when debug state is enabled. / 启用 debug state 时返回 <see langword="true"/>。</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the active TensorRT line does not expose tensor debug-state APIs.
    /// 当前 TensorRT 版本线未暴露 tensor debug-state API 时抛出。
    /// </exception>
    public bool GetTensorDebugState(string tensorName)
    {
        ThrowIfTensorDebugStateUnsupported();
        return NativeBridgeApi.GetExecutionContextTensorDebugState(Line, _handle, tensorName);
    }

    /// <summary>
    /// Enables or disables TensorRT debug state for all tensors on supported execution contexts.
    /// 在受支持的 execution context 上启用或禁用所有 tensor 的 debug state。
    /// </summary>
    /// <param name="enabled">Whether debug state should be enabled. / 是否启用 debug state。</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when the active TensorRT line does not expose tensor debug-state APIs.
    /// 当前 TensorRT 版本线未暴露 tensor debug-state API 时抛出。
    /// </exception>
    public void SetAllTensorsDebugState(bool enabled)
    {
        ThrowIfTensorDebugStateUnsupported();
        NativeBridgeApi.SetAllExecutionContextTensorsDebugState(Line, _handle, enabled);
    }

    private void ThrowIfTensorDebugStateUnsupported()
    {
        if (!SupportsTensorDebugState)
        {
            throw new NotSupportedException("The active TensorRT line does not expose IExecutionContext tensor debug-state APIs. 当前 TensorRT 版本线未暴露 IExecutionContext tensor debug-state API。");
        }
    }
}
