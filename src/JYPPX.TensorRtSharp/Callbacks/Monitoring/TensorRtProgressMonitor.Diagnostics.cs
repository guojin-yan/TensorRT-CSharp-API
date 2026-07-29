using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProgressMonitor
{
    /// <summary>
    /// Synchronously emits a diagnostic progress monitor event through the native monitor object.
    /// 通过 native monitor 对象同步发送一条诊断 progress 事件。
    /// </summary>
    /// <param name="kind">The event kind to emit. 要触发的事件类型。</param>
    /// <param name="phaseName">The phase name copied to native UTF-8 memory. 复制到 native UTF-8 内存的阶段名称。</param>
    /// <param name="parentPhase">The optional parent phase. 可选父阶段。</param>
    /// <param name="step">The completed step index for step-complete events. step-complete 事件的完成步骤索引。</param>
    /// <param name="stepCount">The phase step count for phase-start events. phase-start 事件的总步骤数。</param>
    /// <returns>The diagnostic result reported by the native bridge. native bridge 返回的诊断结果。</returns>
    /// <remarks>
    /// This method is intended for diagnostics and smoke tests. It does not expose the native monitor pointer and does not attach the monitor to a builder config.
    /// 该方法用于诊断和 smoke 测试；它不会暴露 native monitor 指针，也不会把 monitor 绑定到 builder config。
    /// </remarks>
    public TensorRtProgressMonitorDiagnosticResult EmitDiagnostic(
        TensorRtProgressMonitorEventKind kind,
        string phaseName,
        string? parentPhase = null,
        int step = -1,
        int stepCount = 0)
    {
        if (phaseName == null)
        {
            throw new ArgumentNullException(nameof(phaseName));
        }

        ThrowIfDisposed();
        return NativeBridgeApi.EmitProgressMonitorDiagnostic(Line, _handle, kind, phaseName, parentPhase, step, stepCount);
    }
}
