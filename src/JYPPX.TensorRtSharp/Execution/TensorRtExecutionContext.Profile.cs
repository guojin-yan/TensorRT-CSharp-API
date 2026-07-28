using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets or sets the Optimization Profile Index value.
    /// 获取或设置 Optimization Profile Index 值。
    /// </summary>
    public int OptimizationProfileIndex => NativeBridgeApi.GetExecutionContextOptimizationProfile(Line, _handle);

    /// <summary>
    /// Gets or sets the Enqueue Emits Profile value.
    /// 获取或设置 Enqueue Emits Profile 值。
    /// </summary>
    public bool EnqueueEmitsProfile
    {
        get => NativeBridgeApi.GetExecutionContextEnqueueEmitsProfile(Line, _handle);
        set => NativeBridgeApi.SetExecutionContextEnqueueEmitsProfile(Line, _handle, value);
    }

    /// <summary>
    /// Sets the Optimization Profile Async value.
    /// 设置 Optimization Profile Async 值。
    /// </summary>
    public void SetOptimizationProfileAsync(int profileIndex, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.SetExecutionContextOptimizationProfileAsync(Line, _handle, profileIndex, stream.Handle);
    }

    /// <summary>
    /// Attaches a managed TensorRT profiler to this execution context.
    /// 将托管 TensorRT profiler 绑定到当前 execution context。
    /// </summary>
    /// <param name="profiler">The managed profiler to borrow. 要借用的托管 profiler。</param>
    /// <remarks>
    /// TensorRT borrows the native profiler pointer and does not take ownership. This execution context keeps the managed
    /// profiler alive until <see cref="ClearProfiler"/> or <see cref="Dispose"/> detaches it. Dispose the execution context
    /// or clear the profiler before disposing the profiler when possible; if the profiler is disposed first, native release
    /// is deferred until this context detaches it.
    /// TensorRT 只借用 native profiler 指针，不接管所有权。当前 execution context 会保持托管 profiler 存活，直到
    /// <see cref="ClearProfiler"/> 或 <see cref="Dispose"/> 解除绑定。建议先释放 execution context 或清除 profiler 再释放 profiler；
    /// 如果先释放 profiler，native 释放会延迟到 context 解除绑定之后。
    /// </remarks>
    public void SetProfiler(TensorRtProfiler profiler)
    {
        if (profiler == null)
        {
            throw new ArgumentNullException(nameof(profiler));
        }

        profiler.ThrowIfDisposed();
        profiler.AttachBorrower(Line);
        try
        {
            NativeBridgeApi.SetExecutionContextProfiler(Line, _handle, profiler.Handle);
            TensorRtProfiler? previous = _profilerKeepAlive;
            _profilerKeepAlive = profiler;
            previous?.DetachBorrower();
        }
        catch
        {
            profiler.DetachBorrower();
            throw;
        }
    }

    /// <summary>
    /// Reports accumulated execution data to the profiler attached to the context, when one exists.
    /// 将当前执行上下文已累计的执行数据上报给关联的 profiler（如果存在）。
    /// </summary>
    /// <returns>
    /// <see langword="true"/> when TensorRT reports that profiler data was emitted; otherwise <see langword="false"/>.
    /// 当 TensorRT 确认已上报 profiler 数据时返回 <see langword="true"/>，否则返回 <see langword="false"/>。
    /// </returns>
    public bool ReportToProfiler()
    {
        return NativeBridgeApi.ReportExecutionContextToProfiler(Line, _handle);
    }
}
