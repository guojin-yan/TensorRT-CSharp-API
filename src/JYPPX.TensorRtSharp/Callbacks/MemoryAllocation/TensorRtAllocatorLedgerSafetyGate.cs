using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates copied allocator owner ledger diagnostics before any real TensorRT allocator callback runtime proof.
/// 在进入真实 TensorRT allocator callback runtime proof 之前，评估复制出的 allocator owner ledger 诊断。
/// </summary>
/// <remarks>
/// This gate combines the managed owner keep-alive prototype with the native owner state ledger dry-run. It never calls
/// <c>setGpuAllocator</c>, never exposes a native owner handle, device pointer, or CUDA stream handle, and never proves
/// that TensorRT invoked <c>IGpuAllocator</c> or <c>IGpuAsyncAllocator</c> callbacks.
/// 该门禁组合托管 owner keep-alive prototype 与 native owner state ledger dry-run。它不会调用
/// <c>setGpuAllocator</c>，不会暴露 native owner handle、device pointer 或 CUDA stream handle，也不证明 TensorRT
/// 已调用 <c>IGpuAllocator</c> 或 <c>IGpuAsyncAllocator</c> callback。
/// </remarks>
public static class TensorRtAllocatorLedgerSafetyGate
{
    /// <summary>
    /// Runs the allocator ledger safety gate using copied diagnostics only.
    /// 仅使用复制出的诊断运行 allocator ledger 安全门禁。
    /// </summary>
    /// <param name="owner">The managed allocator callback owner. 托管 allocator callback owner。</param>
    /// <param name="line">The TensorRT API line used for the native ledger dry-run. native ledger dry-run 使用的 TensorRT API line。</param>
    /// <param name="request">The copied allocator request used by the diagnostic prototype. 诊断 prototype 使用的复制 allocator 请求。</param>
    /// <param name="targetKind">The copied native owner target kind label. 复制的 native owner 目标类型标签。</param>
    /// <param name="streamValue">The synthetic stream value copied into the ledger. 复制到 ledger 的合成 stream 值。</param>
    /// <returns>A pointer-free safety gate result. 不含 pointer 的安全门禁结果。</returns>
    public static TensorRtAllocatorLedgerSafetyGateResult Evaluate(
        TensorRtAllocatorCallbackOwner owner,
        TensorRtApiLine line,
        TensorRtAllocatorDryRunRequest request,
        string targetKind = "IGpuAllocator",
        ulong streamValue = 0UL)
    {
        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }

        TensorRtAllocatorInternalRuntimePrototypeResult prototype =
            owner.RunInternalSyncAllocatorRuntimePrototype(request);

        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger = null;
        BridgeStatusCode nativeStatus = BridgeStatusCode.Ok;
        string nativeDiagnostic;
        bool nativeAvailable = false;
        try
        {
            nativeLedger = owner.RunNativeStateLedgerDryRunDiagnostic(line, request, targetKind, streamValue);
            nativeStatus = nativeLedger.Value.LastStatus;
            nativeDiagnostic = nativeLedger.Value.Diagnostic;
            nativeAvailable = true;
        }
        catch (Exception exception)
        {
            nativeStatus = BridgeStatusCode.RuntimeError;
            nativeDiagnostic = "allocator-owner-ledger-safety-gate native ledger diagnostic unavailable: " +
                exception.GetType().Name +
                ": " +
                exception.Message;
        }

        return new TensorRtAllocatorLedgerSafetyGateResult(
            line,
            prototype,
            nativeLedger,
            nativeStatus,
            nativeDiagnostic,
            nativeAvailable);
    }

    /// <summary>
    /// Copies the current managed owner lifecycle state without running native ledger diagnostics.
    /// 复制当前托管 owner 生命周期状态，不运行 native ledger 诊断。
    /// </summary>
    /// <param name="owner">The managed allocator callback owner. 托管 allocator callback owner。</param>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A pointer-free safety gate result with native ledger marked unavailable. native ledger 标为不可用的无 pointer 门禁结果。</returns>
    public static TensorRtAllocatorLedgerSafetyGateResult GetSnapshot(
        TensorRtAllocatorCallbackOwner owner,
        string operation = "snapshot")
    {
        return GetSnapshot(owner, TensorRtApiLine.TensorRt11, operation);
    }

    /// <summary>
    /// Copies the current managed owner lifecycle state for an explicit TensorRT API line.
    /// 为显式 TensorRT API 版本线复制当前托管 owner 生命周期状态。
    /// </summary>
    /// <param name="owner">The managed allocator callback owner. 托管 allocator callback owner。</param>
    /// <param name="line">The TensorRT API line represented by the snapshot. snapshot 表示的 TensorRT API 版本线。</param>
    /// <param name="operation">The copied operation label. 复制出的操作标签。</param>
    /// <returns>A pointer-free safety gate result with native ledger marked unavailable. native ledger 标为不可用的无 pointer 门禁结果。</returns>
    public static TensorRtAllocatorLedgerSafetyGateResult GetSnapshot(
        TensorRtAllocatorCallbackOwner owner,
        TensorRtApiLine line,
        string operation = "snapshot")
    {
        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }

        TensorRtAllocatorInternalRuntimePrototypeResult prototype =
            owner.GetInternalRuntimePrototypeSnapshot(operation);
        return new TensorRtAllocatorLedgerSafetyGateResult(
            line,
            prototype,
            null,
            BridgeStatusCode.NotReady,
            "allocator-owner-ledger-safety-gate native ledger was not requested for this copied snapshot.",
            false);
    }
}
