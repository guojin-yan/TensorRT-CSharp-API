using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Clears the CUDA memory address currently bound to a named input or output tensor.
    /// 清除当前绑定到指定输入或输出 tensor 的 CUDA 内存地址。
    /// </summary>
    /// <param name="tensorName">The tensor name. Tensor 名称。</param>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the clear operation.
    /// 当 TensorRT 接受清理操作时返回 <see langword="true"/>。
    /// </returns>
    /// <remarks>
    /// This is a safe high-level wrapper over TensorRT 11 <c>setTensorAddress(name, nullptr)</c>.
    /// 这是 TensorRT 11 <c>setTensorAddress(name, nullptr)</c> 的安全高层封装。
    /// </remarks>
    public bool ClearTensorAddress(string tensorName)
    {
        return NativeBridgeApi.ClearExecutionContextTensorAddress(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the CUDA memory address currently bound to a named input tensor.
    /// 清除当前绑定到指定输入 tensor 的 CUDA 内存地址。
    /// </summary>
    /// <param name="tensorName">The input tensor name. 输入 tensor 名称。</param>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the clear operation.
    /// 当 TensorRT 接受清理操作时返回 <see langword="true"/>。
    /// </returns>
    public bool ClearInputTensorAddress(string tensorName)
    {
        return NativeBridgeApi.ClearExecutionContextInputTensorAddress(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the CUDA memory address currently bound to a named output tensor.
    /// 清除当前绑定到指定输出 tensor 的 CUDA 内存地址。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the clear operation.
    /// 当 TensorRT 接受清理操作时返回 <see langword="true"/>。
    /// </returns>
    public bool ClearOutputTensorAddress(string tensorName)
    {
        return NativeBridgeApi.ClearExecutionContextOutputTensorAddress(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the externally supplied device-memory block for this execution context.
    /// 清除当前 execution context 外部传入的 device memory 块。
    /// </summary>
    /// <remarks>
    /// Use this only after queued inference work has completed. The managed wrapper does not expose the raw native pointer.
    /// 请仅在已提交的推理任务完成后使用。托管封装不会向普通用户暴露原生裸指针。
    /// </remarks>
    public void ClearDeviceMemory()
    {
        NativeBridgeApi.ClearExecutionContextDeviceMemory(Line, _handle);
    }

    /// <summary>
    /// Clears the CUDA event used by TensorRT to signal input-consumption completion.
    /// 清除 TensorRT 用于通知输入消费完成的 CUDA event。
    /// </summary>
    /// <returns>
    /// <see langword="true"/> when TensorRT accepted the clear operation.
    /// 当 TensorRT 接受清理操作时返回 <see langword="true"/>。
    /// </returns>
    public bool ClearInputConsumedEvent()
    {
        return NativeBridgeApi.ClearExecutionContextInputConsumedEvent(Line, _handle);
    }

    /// <summary>
    /// Sets auxiliary CUDA streams used by TensorRT 8, 10, or 11 during inference.
    /// 设置 TensorRT 8、10 或 11 推理阶段使用的辅助 CUDA stream。
    /// </summary>
    /// <param name="streams">
    /// Caller-owned auxiliary streams. The context keeps their native handles alive while TensorRT may borrow them.
    /// 调用方拥有的辅助 stream；在 TensorRT 可能借用期间，context 会保持其 native handle 存活。
    /// </param>
    /// <remarks>
    /// Passing an empty collection clears user-provided auxiliary streams. TensorRT may still use its default internal behavior.
    /// 传入空集合会清除用户提供的辅助 stream；TensorRT 仍可能使用自身默认内部行为。
    /// </remarks>
    public void SetAuxStreams(IReadOnlyList<CudaStream> streams)
    {
        if (streams == null)
        {
            throw new ArgumentNullException(nameof(streams));
        }

        if (streams.Count == 0)
        {
            ClearAuxStreams();
            return;
        }

        SafeCudaStreamHandle[] handles = new SafeCudaStreamHandle[streams.Count];
        for (int i = 0; i < streams.Count; i++)
        {
            CudaStream stream = streams[i];
            if (stream == null)
            {
                throw new ArgumentException("Auxiliary stream collection must not contain null entries.", nameof(streams));
            }

            handles[i] = stream.Handle;
        }

        TensorRtAuxiliaryStreamHandleLease? pendingLease = TensorRtAuxiliaryStreamHandleLease.Create(handles);
        try
        {
            lock (_auxiliaryStreamLeaseLock)
            {
                ThrowIfAuxiliaryStreamContextDisposed();
                NativeBridgeApi.SetExecutionContextAuxStreams(Line, _handle, pendingLease.Handles);

                TensorRtAuxiliaryStreamHandleLease? previousLease = _auxiliaryStreamLease;
                _auxiliaryStreamLease = pendingLease;
                pendingLease = null;
                _auxiliaryStreamAssignedCount = handles.Length;
                _auxiliaryStreamsCleared = false;
                _auxiliaryStreamDiagnostic = $"{handles.Length} caller-provided auxiliary CUDA stream handle lease(s) are active.";
                previousLease?.Dispose();
            }
        }
        finally
        {
            pendingLease?.Dispose();
        }
    }

    /// <summary>
    /// Sets auxiliary CUDA streams used by TensorRT 8, 10, or 11 during inference.
    /// 设置 TensorRT 8、10 或 11 推理阶段使用的辅助 CUDA stream。
    /// </summary>
    /// <param name="streams">Auxiliary CUDA streams. 辅助 CUDA stream。</param>
    public void SetAuxStreams(params CudaStream[] streams)
    {
        if (streams == null)
        {
            throw new ArgumentNullException(nameof(streams));
        }

        SetAuxStreams((IReadOnlyList<CudaStream>)streams);
    }

    /// <summary>
    /// Gets a pointer-free snapshot of auxiliary-stream assignment and managed handle-lease state.
    /// 获取不含指针的辅助 stream 分配与托管 handle lease 状态快照。
    /// </summary>
    public TensorRtAuxiliaryStreamAssignmentSnapshot GetAuxiliaryStreamAssignmentSnapshot()
    {
        lock (_auxiliaryStreamLeaseLock)
        {
            return new TensorRtAuxiliaryStreamAssignmentSnapshot(
                Line,
                _auxiliaryStreamAssignedCount,
                _auxiliaryStreamsCleared,
                _auxiliaryStreamLease != null,
                _auxiliaryStreamDiagnostic);
        }
    }

    private void ThrowIfAuxiliaryStreamContextDisposed()
    {
        if (_auxiliaryStreamContextDisposed || _handle.IsClosed || _handle.IsInvalid)
        {
            throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
        }
    }
}
