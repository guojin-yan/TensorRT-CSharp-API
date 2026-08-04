using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Sets the TensorRT builder optimization level.
    /// 设置 TensorRT builder 优化级别。
    /// </summary>
    /// <param name="level">The optimization level. 优化级别。</param>
    public void SetOptimizationLevel(int level)
    {
        NativeBridgeApi.SetBuilderOptimizationLevel(Line, _handle, level);
    }

    /// <summary>
    /// Gets the TensorRT builder optimization level.
    /// 获取 TensorRT builder 优化级别。
    /// </summary>
    /// <returns>The current optimization level. 当前优化级别。</returns>
    public int GetOptimizationLevel()
    {
        return NativeBridgeApi.GetBuilderOptimizationLevel(Line, _handle);
    }

    /// <summary>
    /// Sets TensorRT profiling verbosity for build diagnostics.
    /// 设置构建诊断使用的 TensorRT profiling verbosity。
    /// </summary>
    /// <param name="verbosity">The profiling verbosity. profiling 详细程度。</param>
    public void SetProfilingVerbosity(TensorRtProfilingVerbosity verbosity)
    {
        NativeBridgeApi.SetProfilingVerbosity(Line, _handle, verbosity);
    }

    /// <summary>
    /// Gets TensorRT profiling verbosity for this builder configuration.
    /// 获取当前 builder 配置的 TensorRT profiling verbosity。
    /// </summary>
    /// <returns>The configured profiling verbosity. 已配置的 profiling verbosity。</returns>
    public TensorRtProfilingVerbosity GetProfilingVerbosity()
    {
        return NativeBridgeApi.GetProfilingVerbosity(Line, _handle);
    }

    /// <summary>
    /// Sets the maximum auxiliary CUDA stream count TensorRT may use.
    /// 设置 TensorRT 可使用的最大辅助 CUDA stream 数量。
    /// </summary>
    /// <param name="maxStreams">The maximum auxiliary stream count. 最大辅助 stream 数量。</param>
    public void SetMaxAuxStreams(int maxStreams)
    {
        NativeBridgeApi.SetMaxAuxStreams(Line, _handle, maxStreams);
    }

    /// <summary>
    /// Gets the maximum auxiliary CUDA stream count TensorRT may use.
    /// 获取 TensorRT 可使用的最大辅助 CUDA stream 数量。
    /// </summary>
    /// <returns>The maximum auxiliary stream count. 最大辅助 stream 数量。</returns>
    public int GetMaxAuxStreams()
    {
        return NativeBridgeApi.GetMaxAuxStreams(Line, _handle);
    }

    /// <summary>
    /// Sets the average timing-iteration count used by TensorRT tactic benchmarking.
    /// 设置 TensorRT tactic 基准测试使用的平均 timing 迭代次数。
    /// </summary>
    /// <param name="iterations">The average timing-iteration count. 平均 timing 迭代次数。</param>
    public void SetAverageTimingIterations(int iterations)
    {
        NativeBridgeApi.SetAverageTimingIterations(Line, _handle, iterations);
    }

    /// <summary>
    /// Gets the average timing-iteration count used by TensorRT tactic benchmarking.
    /// 获取 TensorRT tactic 基准测试使用的平均 timing 迭代次数。
    /// </summary>
    /// <returns>The average timing-iteration count. 平均 timing 迭代次数。</returns>
    public int GetAverageTimingIterations()
    {
        return NativeBridgeApi.GetAverageTimingIterations(Line, _handle);
    }

    /// <summary>
    /// Gets TensorRT 8's legacy maximum workspace-size setting.
    /// 获取 TensorRT 8 legacy 最大 workspace size 设置。
    /// </summary>
    /// <remarks>
    /// This is a TensorRT 8 compatibility diagnostic for the deprecated <c>IBuilderConfig::getMaxWorkspaceSize</c> API.
    /// Prefer <see cref="GetMemoryPoolLimit"/> with <see cref="TensorRtMemoryPoolType.Workspace"/> for portable TensorRT 8/10/11 code.
    /// 这是 TensorRT 8 兼容诊断，用于 deprecated <c>IBuilderConfig::getMaxWorkspaceSize</c> 接口。跨版本代码请优先使用
    /// <see cref="GetMemoryPoolLimit"/> 和 <see cref="TensorRtMemoryPoolType.Workspace"/>。
    /// </remarks>
    public ulong MaxWorkspaceSizeCompatibilityInBytes => NativeBridgeApi.GetMaxWorkspaceSizeCompatibility(Line, _handle);

    /// <summary>
    /// Sets the deprecated TensorRT 8 workspace limit in bytes.
    /// 设置已弃用的 TensorRT 8 workspace 字节上限。
    /// </summary>
    /// <remarks>
    /// Prefer <see cref="SetMemoryPoolLimit"/> with <see cref="TensorRtMemoryPoolType.Workspace"/> in portable code.
    /// 跨版本代码请优先使用 <see cref="SetMemoryPoolLimit"/> 与 <see cref="TensorRtMemoryPoolType.Workspace"/>。
    /// </remarks>
    public void SetMaxWorkspaceSizeCompatibility(ulong workspaceSizeInBytes)
    {
        NativeBridgeApi.SetMaxWorkspaceSizeCompatibility(Line, _handle, workspaceSizeInBytes);
    }

    /// <summary>
    /// Gets TensorRT 8's legacy minimum timing-iteration count.
    /// 获取 TensorRT 8 legacy 最小 timing 迭代次数。
    /// </summary>
    /// <remarks>
    /// This is a TensorRT 8 compatibility diagnostic for the deprecated <c>IBuilderConfig::getMinTimingIterations</c> API.
    /// Prefer <see cref="GetAverageTimingIterations"/> for portable TensorRT 8/10/11 timing diagnostics.
    /// 这是 TensorRT 8 兼容诊断，用于 deprecated <c>IBuilderConfig::getMinTimingIterations</c> 接口。跨版本 timing 诊断请优先使用
    /// <see cref="GetAverageTimingIterations"/>。
    /// </remarks>
    public int MinTimingIterationsCompatibility => NativeBridgeApi.GetMinTimingIterationsCompatibility(Line, _handle);

    /// <summary>
    /// Sets the deprecated TensorRT 8 minimum timing iteration count.
    /// 设置已弃用的 TensorRT 8 minimum timing iteration 次数。
    /// </summary>
    /// <remarks>
    /// Prefer <see cref="SetAverageTimingIterations"/> in portable code. TensorRT 10 and 11 report this method as unsupported.
    /// 跨版本代码请优先使用 <see cref="SetAverageTimingIterations"/>；TensorRT 10/11 会将此方法报告为不支持。
    /// </remarks>
    public void SetMinTimingIterationsCompatibility(int iterations)
    {
        NativeBridgeApi.SetMinTimingIterationsCompatibility(Line, _handle, iterations);
    }

    /// <summary>
    /// Sets the TensorRT tactic-source mask.
    /// 设置 TensorRT tactic source 掩码。
    /// </summary>
    /// <param name="sources">The enabled tactic sources. 已启用的 tactic sources。</param>
    public void SetTacticSources(TensorRtTacticSources sources)
    {
        NativeBridgeApi.SetTacticSources(Line, _handle, sources);
    }

    /// <summary>
    /// Gets the TensorRT tactic-source mask.
    /// 获取 TensorRT tactic source 掩码。
    /// </summary>
    /// <returns>The enabled tactic sources. 已启用的 tactic sources。</returns>
    public TensorRtTacticSources GetTacticSources()
    {
        return NativeBridgeApi.GetTacticSources(Line, _handle);
    }

}
