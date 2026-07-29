using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProfiler
{
    /// <summary>
    /// Synchronously emits a diagnostic profiler record through the native profiler object.
    /// 通过 native profiler 对象同步发送一条诊断 profiler 记录。
    /// </summary>
    /// <param name="layerName">The layer name copied to native UTF-8 memory. 复制到 native UTF-8 内存的 layer 名称。</param>
    /// <param name="milliseconds">The diagnostic elapsed time in milliseconds. 诊断耗时，单位毫秒。</param>
    /// <returns><see langword="true"/> when the callback completed without a managed exception or non-OK status. 回调未发生托管异常或非 OK 状态时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native profiler pointer and does not attach the profiler to an execution context.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native profiler 指针，也不会把 profiler 绑定到 execution context。
    /// </remarks>
    public bool EmitDiagnostic(string layerName, float milliseconds)
    {
        if (layerName == null)
        {
            throw new ArgumentNullException(nameof(layerName));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitProfilerDiagnostic(Line, _handle, layerName, milliseconds);
    }
}
