using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Wraps TensorRT <c>IRuntimeConfig</c> for execution-context creation.
/// 封装 TensorRT <c>IRuntimeConfig</c>，用于创建 execution context。
/// </summary>
/// <remarks>
/// This wrapper is currently supported for TensorRT 10 and TensorRT 11 runtime-config paths.
/// 当前封装支持 TensorRT 10 和 TensorRT 11 的 runtime-config 路径。
/// </remarks>
public sealed class TensorRtRuntimeConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtRuntimeConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line that owns this runtime config.
    /// 获取拥有当前 runtime config 的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the execution-context allocation strategy.
    /// 获取或设置 execution context 的内存分配策略。
    /// </summary>
    public TensorRtExecutionContextAllocationStrategy AllocationStrategy
    {
        get => NativeBridgeApi.GetRuntimeConfigAllocationStrategy(Line, _handle);
        set => NativeBridgeApi.SetRuntimeConfigAllocationStrategy(Line, _handle, value);
    }

    /// <summary>
    /// Gets a compact copied summary of this runtime config.
    /// 获取当前 runtime config 的紧凑复制型摘要。
    /// </summary>
    /// <remarks>
    /// This method reads scalar runtime-config state only. It does not expose the native config handle and does not promote runtime proof.
    /// 该方法只读取 runtime-config 标量状态；不暴露原生 config handle，也不会晋级 runtime proof。
    /// </remarks>
    public TensorRtRuntimeConfigSummary ToSummary()
    {
        return new TensorRtRuntimeConfigSummary(Line, AllocationStrategy);
    }

    /// <summary>
    /// Releases the native runtime config handle.
    /// 释放原生 runtime config 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

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
