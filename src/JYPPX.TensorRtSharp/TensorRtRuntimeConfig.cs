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
    /// Releases the native runtime config handle.
    /// 释放原生 runtime config 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
