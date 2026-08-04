using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
    /// <summary>
    /// Runs the synthetic native owner state ledger dry-run sequence.
    /// 执行 synthetic native owner 状态 ledger dry-run 序列。
    /// </summary>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <param name="targetKind">The copied target kind label. 复制的目标类型标签。</param>
    /// <param name="streamValue">The synthetic stream value copied into the ledger. 复制到 ledger 的合成 stream value。</param>
    /// <returns>A copied native owner state snapshot with no handle or device pointer. 不包含 handle 或 device pointer 的 native owner 状态副本。</returns>
    /// <remarks>
    /// This method creates a short-lived native diagnostic owner, records synthetic attach/allocation/release/detach
    /// intents, copies the final state snapshot, and immediately releases the native handle. It does not attach to
    /// TensorRT, does not call <c>setGpuAllocator</c>, and does not implement any allocator callback trampoline.
    /// 该方法会创建短生命周期 native 诊断 owner，记录合成的 attach/allocation/release/detach intent，复制最终状态快照，
    /// 然后立即释放 native 句柄。它不会绑定 TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现任何 allocator callback trampoline。
    /// </remarks>
    public TensorRtAllocatorOwnerStateDryRunResult RunNativeStateLedgerDryRunDiagnostic(
        TensorRtApiLine line,
        TensorRtAllocatorDryRunRequest request,
        string targetKind = "IGpuAllocator",
        ulong streamValue = 0)
    {
        ThrowIfDisposed();

        using SafeTensorRtObjectHandle nativeOwner = NativeBridgeApi.CreateAllocatorOwnerDryRun(line);
        NativeTensorRtAllocatorOwnerStateInfo state = NativeBridgeApi.GetAllocatorOwnerDryRunState(line, nativeOwner);
        state = NativeBridgeApi.AttachAllocatorOwnerDryRunIntent(line, nativeOwner, targetKind);
        state = NativeBridgeApi.RecordAllocatorOwnerDryRunAllocationIntent(line, nativeOwner, request.Size, request.Alignment, streamValue);
        state = NativeBridgeApi.RecordAllocatorOwnerDryRunReleaseIntent(line, nativeOwner, state.LastAllocationId, streamValue);
        state = NativeBridgeApi.DetachAllocatorOwnerDryRunIntent(line, nativeOwner, targetKind);
        return CreateStateDryRunResult(state);
    }

}
