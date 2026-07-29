using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Builds a pointer-free callback owner closure matrix across TensorRT callback families.
/// 构建不暴露裸指针的 TensorRT callback owner 闭环矩阵。
/// </summary>
/// <remarks>
/// The matrix consumes copied diagnostics from existing gates. It does not attach callbacks, install native vtables,
/// allocate device memory, invoke stream read/write callbacks, or promote deferred rows to runtime proof.
/// 该矩阵只消费已有 gate 复制出的诊断，不 attach callback、不安装 native vtable、不分配 device memory、
/// 不调用 stream read/write callback，也不把 deferred 行提升为 runtime proof。
/// </remarks>
public static partial class TensorRtCallbackOwnerClosureMatrix
{
    /// <summary>
    /// Aggregates copied callback owner evidence into a family-level closure matrix.
    /// 将复制出的 callback owner evidence 聚合为按 family 展开的闭环矩阵。
    /// </summary>
    /// <param name="allocatorLedgerSafetyGate">Copied IGpuAllocator ledger gate evidence. 复制出的 IGpuAllocator ledger gate 证据。</param>
    /// <param name="outputAllocatorRuntimeProofPrecheck">Copied OutputAllocator runtime proof precheck evidence. 复制出的 OutputAllocator precheck 证据。</param>
    /// <param name="debugListenerRuntimeProofPrecheck">Copied DebugListener runtime proof precheck evidence. 复制出的 DebugListener precheck 证据。</param>
    /// <param name="streamIoInterfaceInfoDesignGate">Copied stream reader/writer design gate evidence. 复制出的 stream reader/writer design gate 证据。</param>
    /// <returns>A pointer-free closure matrix. 不暴露裸指针的闭环矩阵。</returns>
    public static TensorRtCallbackOwnerClosureMatrixResult Evaluate(
        TensorRtAllocatorLedgerSafetyGateResult allocatorLedgerSafetyGate,
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputAllocatorRuntimeProofPrecheck,
        TensorRtDebugListenerRuntimeProofPrecheckResult debugListenerRuntimeProofPrecheck,
        TensorRtStreamIoInterfaceInfoDesignGateResult streamIoInterfaceInfoDesignGate)
    {
        TensorRtCallbackOwnerClosureMatrixRow[] rows =
        {
            BuildGpuAllocatorRow(allocatorLedgerSafetyGate),
            BuildGpuAsyncAllocatorRow(allocatorLedgerSafetyGate),
            BuildOutputAllocatorRow(outputAllocatorRuntimeProofPrecheck),
            BuildDebugListenerRow(debugListenerRuntimeProofPrecheck),
            BuildStreamReaderWriterRow(streamIoInterfaceInfoDesignGate)
        };

        return new TensorRtCallbackOwnerClosureMatrixResult(rows, BuildMatrixBlockers(rows));
    }

}
