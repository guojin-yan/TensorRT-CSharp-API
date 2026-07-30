using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Compact pointer-free summary of TensorRT runtime-config state.
/// TensorRT runtime-config 状态的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtRuntimeConfigSummary
{
    internal TensorRtRuntimeConfigSummary(TensorRtApiLine line, TensorRtExecutionContextAllocationStrategy allocationStrategy)
    {
        Line = line;
        AllocationStrategy = allocationStrategy;
    }

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets copied execution-context allocation strategy. 获取已复制 execution-context allocation strategy。</summary>
    public TensorRtExecutionContextAllocationStrategy AllocationStrategy { get; }

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether deferred records can be deleted because of this summary. 获取是否可因该摘要删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for smoke output and logs. 将该摘要格式化为 smoke 输出和日志。</summary>
    public override string ToString() => $"Line={(int)Line} AllocationStrategy={AllocationStrategy} RuntimeProof={CanPromoteRuntimeProof}";
}
