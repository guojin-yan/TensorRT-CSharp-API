using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns a bridge wrapper for a CUDA device primary execution context.
/// 拥有 CUDA 设备主执行上下文的 bridge 包装器。
/// </summary>
/// <remarks>
/// Disposing this object releases only the bridge wrapper. The CUDA primary context is device-owned
/// and is never passed to <c>cudaExecutionCtxDestroy</c>.
/// 释放此对象只会释放 bridge 包装器；CUDA 主上下文由设备拥有，绝不会传给
/// <c>cudaExecutionCtxDestroy</c>。
/// </remarks>
public sealed class CudaPrimaryExecutionContext : IDisposable
{
    private readonly SafeCudaExecutionContextHandle _handle;

    internal CudaPrimaryExecutionContext(SafeCudaExecutionContextHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    /// <summary>Gets whether this wrapper represents the device primary context. 获取此包装器是否表示设备主上下文。</summary>
    public bool IsPrimary => true;

    /// <summary>Gets the device ordinal reported by CUDA for this context. 获取 CUDA 为此上下文报告的设备序号。</summary>
    public int DeviceOrdinal => NativeCudaApi.GetExecutionContextDevice(_handle);

    /// <summary>Gets the process-unique CUDA execution-context id. 获取进程内唯一的 CUDA 执行上下文 ID。</summary>
    public ulong Id => NativeCudaApi.GetExecutionContextId(_handle);

    /// <summary>Blocks until work tracked by this execution context completes. 阻塞直到此执行上下文跟踪的工作完成。</summary>
    public void Synchronize()
    {
        NativeCudaApi.SynchronizeExecutionContext(_handle);
    }

    /// <summary>Creates a stream explicitly bound to this primary execution context. 创建显式绑定到此主执行上下文的 stream。</summary>
    /// <param name="flags">CUDA stream creation flags. CUDA stream 创建标志。</param>
    /// <param name="priority">CUDA stream priority. CUDA stream 优先级。</param>
    public CudaStream CreateStream(
        CudaStreamCreationFlags flags = CudaStreamCreationFlags.Default,
        int priority = 0)
    {
        return new CudaStream(NativeCudaApi.CreateExecutionContextStream(_handle, flags, priority));
    }

    /// <summary>Records all currently tracked context work into an event. 将当前上下文已跟踪的工作记录到 event。</summary>
    /// <param name="cudaEvent">The bridge-owned CUDA event. Bridge 拥有的 CUDA event。</param>
    public void RecordEvent(CudaEvent cudaEvent)
    {
        if (cudaEvent == null) throw new ArgumentNullException(nameof(cudaEvent));
        NativeCudaApi.RecordExecutionContextEvent(_handle, cudaEvent.Handle);
    }

    /// <summary>Makes future context work wait for an event without blocking the CPU. 让后续上下文工作等待 event，且不阻塞 CPU。</summary>
    /// <param name="cudaEvent">The bridge-owned CUDA event. Bridge 拥有的 CUDA event。</param>
    public void WaitEvent(CudaEvent cudaEvent)
    {
        if (cudaEvent == null) throw new ArgumentNullException(nameof(cudaEvent));
        NativeCudaApi.WaitExecutionContextEvent(_handle, cudaEvent.Handle);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
