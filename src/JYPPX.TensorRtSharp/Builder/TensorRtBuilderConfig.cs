using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private TensorRtProgressMonitor? _progressMonitorKeepAlive;
    private bool _disposed;

    internal TensorRtBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this builder configuration.
    /// 获取当前 builder 配置使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Adds an optimization profile to this builder configuration.
    /// 向当前 builder 配置添加一个 optimization profile。
    /// </summary>
    /// <param name="profile">The optimization profile to attach. 要附加的 optimization profile。</param>
    /// <returns>The zero-based profile index assigned by TensorRT. TensorRT 分配的从零开始的 profile 索引。</returns>
    public int AddOptimizationProfile(TensorRtOptimizationProfile profile)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (profile.Line != Line)
        {
            throw new ArgumentException("Optimization profile must belong to the same TensorRT API line as the builder config.");
        }

        return NativeBridgeApi.AddOptimizationProfile(Line, _handle, profile.Handle);
    }

    /// <summary>
    /// Gets the number of optimization profiles currently attached to this builder config.
    /// 获取当前 builder config 已附加的 optimization profile 数量。
    /// </summary>
    public int OptimizationProfileCount => NativeBridgeApi.GetBuilderConfigOptimizationProfileCount(Line, _handle);

    /// <summary>
    /// Sets the CUDA stream TensorRT uses for profiling work during engine building.
    /// 设置 TensorRT 在 engine 构建 profiling 阶段使用的 CUDA stream。
    /// </summary>
    /// <param name="stream">The CUDA stream used by TensorRT profiling. TensorRT profiling 使用的 CUDA stream。</param>
    /// <remarks>
    /// The stream must remain alive until the build operation that uses this config has completed.
    /// 该 stream 必须至少存活到使用此 config 的构建操作结束。
    /// </remarks>
    public void SetProfileStream(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.SetBuilderConfigProfileStream(Line, _handle, stream.Handle);
    }

    /// <summary>
    /// Returns whether this builder config has a profiling CUDA stream set.
    /// 返回当前 builder config 是否已经设置 profiling CUDA stream。
    /// </summary>
    public bool IsProfileStreamSet => NativeBridgeApi.IsBuilderConfigProfileStreamSet(Line, _handle);

    /// <summary>
    /// Sets the optimization profile TensorRT should use for INT8 calibration.
    /// 设置 TensorRT 在 INT8 calibration 中使用的 optimization profile。
    /// </summary>
    /// <param name="profile">The calibration optimization profile. 用于 calibration 的 optimization profile。</param>
    /// <remarks>
    /// TensorRT 10 keeps this API for compatibility but marks it deprecated upstream; it is still useful for TensorRT 8/10 deployment migration.
    /// TensorRT 10 上游已将该 API 标记为 deprecated，但它对 TensorRT 8/10 部署迁移仍有实际价值。
    /// </remarks>
    public void SetCalibrationProfile(TensorRtOptimizationProfile profile)
    {
        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        if (profile.Line != Line)
        {
            throw new ArgumentException("Calibration profile must belong to the same TensorRT API line as the builder config.");
        }

        NativeBridgeApi.SetBuilderConfigCalibrationProfile(Line, _handle, profile.Handle);
    }

    /// <summary>
    /// Returns whether TensorRT currently has a calibration profile attached to this builder config.
    /// 返回 TensorRT 当前是否为此 builder config 附加了 calibration profile。
    /// </summary>
    public bool HasCalibrationProfile => NativeBridgeApi.HasBuilderConfigCalibrationProfile(Line, _handle);

    /// <summary>
    /// Returns whether TensorRT currently has an algorithm selector attached, without exposing the borrowed selector pointer.
    /// 返回 TensorRT 当前是否附加了 algorithm selector；该属性只报告 presence，不暴露 borrowed selector 指针。
    /// </summary>
    /// <remarks>
    /// This TensorRT 8/10 compatibility probe does not transfer ownership and cannot be used to invoke selector callbacks.
    /// 这是 TensorRT 8/10 compatibility 查询，不转移生命周期，也不能用于调用 selector 回调。
    /// </remarks>
    public bool HasAlgorithmSelectorCompatibility => NativeBridgeApi.HasBuilderConfigAlgorithmSelectorCompatibility(Line, _handle);

    /// <summary>
    /// Returns whether TensorRT currently has an INT8 calibrator attached, without exposing the borrowed calibrator pointer.
    /// 返回 TensorRT 当前是否附加了 INT8 calibrator；该属性只报告 presence，不暴露 borrowed calibrator 指针。
    /// </summary>
    /// <remarks>
    /// This TensorRT 8/10 compatibility probe does not transfer ownership and cannot be used to invoke calibrator callbacks.
    /// 这是 TensorRT 8/10 compatibility 查询，不转移生命周期，也不能用于调用 calibrator 回调。
    /// </remarks>
    public bool HasInt8CalibratorCompatibility => NativeBridgeApi.HasBuilderConfigInt8CalibratorCompatibility(Line, _handle);

    /// <summary>
    /// Enables or disables one TensorRT builder flag.
    /// 启用或禁用一个 TensorRT builder 标志。
    /// </summary>
    /// <param name="flag">The builder flag to change. 要修改的 builder 标志。</param>
    /// <param name="enabled">Whether the flag should be enabled. 是否启用该标志。</param>
    public void SetFlag(TensorRtBuilderFlag flag, bool enabled = true)
    {
        NativeBridgeApi.SetBuilderConfigFlag(Line, _handle, flag, enabled);
    }

    /// <summary>
    /// Clears one TensorRT builder flag.
    /// 清除一个 TensorRT builder 标志。
    /// </summary>
    /// <param name="flag">The builder flag to clear. 要清除的 builder 标志。</param>
    public void ClearFlag(TensorRtBuilderFlag flag)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            NativeBridgeApi.ClearBuilderConfigFlag(Line, _handle, flag);
            return;
        }

        SetFlag(flag, false);
    }

    /// <summary>
    /// Returns whether one TensorRT builder flag is currently enabled.
    /// 返回某个 TensorRT builder 标志当前是否启用。
    /// </summary>
    /// <param name="flag">The builder flag to query. 要查询的 builder 标志。</param>
    /// <returns><see langword="true"/> when the flag is enabled. 当该标志已启用时返回 <see langword="true"/>。</returns>
    public bool GetFlag(TensorRtBuilderFlag flag)
    {
        return NativeBridgeApi.GetBuilderConfigFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Sets the TensorRT engine capability used by the builder config.
    /// 设置 builder config 使用的 TensorRT engine capability。
    /// </summary>
    /// <param name="capability">The target engine capability. 目标 engine capability。</param>
    public void SetEngineCapability(TensorRtEngineCapability capability)
    {
        NativeBridgeApi.SetBuilderConfigEngineCapability(Line, _handle, capability);
    }

    /// <summary>
    /// Gets the TensorRT engine capability currently configured for this builder config.
    /// 获取当前 builder config 中配置的 TensorRT engine capability。
    /// </summary>
    /// <returns>The configured engine capability. 已配置的 engine capability。</returns>
    public TensorRtEngineCapability GetEngineCapability()
    {
        return NativeBridgeApi.GetBuilderConfigEngineCapability(Line, _handle);
    }

    /// <summary>
    /// Enables or disables a TensorRT preview feature.
    /// 启用或禁用 TensorRT 预览特性。
    /// </summary>
    /// <param name="feature">The preview feature. 预览特性。</param>
    /// <param name="enabled">Whether the feature should be enabled. 是否启用该特性。</param>
    public void SetPreviewFeature(TensorRtPreviewFeature feature, bool enabled)
    {
        NativeBridgeApi.SetBuilderConfigPreviewFeature(Line, _handle, feature, enabled);
    }

    /// <summary>
    /// Queries whether a TensorRT preview feature is enabled.
    /// 查询某个 TensorRT 预览特性是否已启用。
    /// </summary>
    /// <param name="feature">The preview feature. 预览特性。</param>
    /// <returns><c>true</c> when the preview feature is enabled. 如果该预览特性已启用，则返回 <c>true</c>。</returns>
    public bool GetPreviewFeature(TensorRtPreviewFeature feature)
    {
        return NativeBridgeApi.GetBuilderConfigPreviewFeature(Line, _handle, feature);
    }

    /// <summary>
    /// Sets the hardware compatibility level for generated TensorRT engines.
    /// 设置生成 TensorRT engine 时使用的硬件兼容性级别。
    /// </summary>
    /// <param name="level">The hardware compatibility level. 硬件兼容性级别。</param>
    public void SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel level)
    {
        NativeBridgeApi.SetBuilderConfigHardwareCompatibilityLevel(Line, _handle, level);
    }

    /// <summary>
    /// Gets the configured TensorRT hardware compatibility level.
    /// 获取当前配置的 TensorRT 硬件兼容性级别。
    /// </summary>
    /// <returns>The configured hardware compatibility level. 已配置的硬件兼容性级别。</returns>
    public TensorRtHardwareCompatibilityLevel GetHardwareCompatibilityLevel()
    {
        return NativeBridgeApi.GetBuilderConfigHardwareCompatibilityLevel(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 10.x runtime platform for cross-platform engine generation.
    /// 设置 TensorRT 10.x 用于跨平台 engine 生成的 runtime platform。
    /// </summary>
    /// <param name="platform">The TensorRT runtime platform. TensorRT 运行时平台。</param>
    /// <remarks>
    /// TensorRT 8.x does not expose this option and the native bridge returns NotSupported.
    /// TensorRT 8.x 不暴露该选项，原生桥接会返回 NotSupported。
    /// </remarks>
    public void SetRuntimePlatform(TensorRtRuntimePlatform platform)
    {
        NativeBridgeApi.SetBuilderConfigRuntimePlatform(Line, _handle, platform);
    }

    /// <summary>
    /// Gets the TensorRT 10.x runtime platform configured for this builder config.
    /// 获取当前 builder config 配置的 TensorRT 10.x runtime platform。
    /// </summary>
    /// <returns>The configured runtime platform. 已配置的 runtime platform。</returns>
    /// <remarks>
    /// TensorRT 8.x does not expose this option and the native bridge returns NotSupported.
    /// TensorRT 8.x 不暴露该选项，原生桥接会返回 NotSupported。
    /// </remarks>
    public TensorRtRuntimePlatform GetRuntimePlatform()
    {
        return NativeBridgeApi.GetBuilderConfigRuntimePlatform(Line, _handle);
    }

    /// <summary>
    /// Pins one layer to a specific TensorRT device type.
    /// 将一个 layer 固定到指定的 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <param name="deviceType">The TensorRT device type. TensorRT 设备类型。</param>
    public void SetLayerDeviceType(TensorRtLayer layer, TensorRtDeviceType deviceType)
    {
        ValidateLayer(layer);
        NativeBridgeApi.SetLayerDeviceType(Line, _handle, layer.Handle, deviceType);
    }

    /// <summary>
    /// Gets the TensorRT device type assigned to one layer.
    /// 获取一个 layer 当前分配到的 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <returns>The configured TensorRT device type. 已配置的 TensorRT device type。</returns>
    public TensorRtDeviceType GetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.GetLayerDeviceType(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Returns whether one layer has an explicit TensorRT device type.
    /// 返回一个 layer 是否具有显式 TensorRT device type。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    /// <returns><see langword="true"/> when the layer has an explicit assignment. 当该 layer 具有显式分配时返回 <see langword="true"/>。</returns>
    public bool IsLayerDeviceTypeSet(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.IsLayerDeviceTypeSet(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Removes the explicit TensorRT device-type assignment for one layer.
    /// 移除一个 layer 的显式 TensorRT device type 分配。
    /// </summary>
    /// <param name="layer">The target layer. 目标 layer。</param>
    public void ResetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        NativeBridgeApi.ResetLayerDeviceType(Line, _handle, layer.Handle);
    }

    /// <summary>
    /// Sets a TensorRT memory-pool size limit.
    /// 设置一个 TensorRT memory pool 大小上限。
    /// </summary>
    /// <param name="pool">The TensorRT memory pool. TensorRT 内存池。</param>
    /// <param name="bytes">The size limit in bytes. 大小上限，单位为字节。</param>
    public void SetMemoryPoolLimit(TensorRtMemoryPoolType pool, ulong bytes)
    {
        NativeBridgeApi.SetMemoryPoolLimit(Line, _handle, pool, bytes);
    }

    /// <summary>
    /// Gets a TensorRT memory-pool size limit.
    /// 获取一个 TensorRT memory pool 大小上限。
    /// </summary>
    /// <param name="pool">The TensorRT memory pool. TensorRT 内存池。</param>
    /// <returns>The size limit in bytes. 大小上限，单位为字节。</returns>
    public ulong GetMemoryPoolLimit(TensorRtMemoryPoolType pool)
    {
        return NativeBridgeApi.GetMemoryPoolLimit(Line, _handle, pool);
    }

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

    /// <summary>
    /// Creates a TensorRT timing cache from optional serialized bytes.
    /// 使用可选的序列化字节创建一个 TensorRT timing cache。
    /// </summary>
    /// <param name="serializedCache">Optional serialized timing-cache payload. 可选的序列化 timing cache 负载。</param>
    /// <returns>A TensorRT timing-cache wrapper. TensorRT timing cache 封装。</returns>
    public TensorRtTimingCache CreateTimingCache(byte[]? serializedCache = null)
    {
        return new TensorRtTimingCache(Line, NativeBridgeApi.CreateTimingCache(Line, _handle, serializedCache));
    }

    /// <summary>
    /// Attaches a TensorRT timing cache to this builder configuration.
    /// 将一个 TensorRT timing cache 附加到当前 builder 配置。
    /// </summary>
    /// <param name="cache">The timing cache to attach. 要附加的 timing cache。</param>
    /// <param name="ignoreMismatch">Whether TensorRT should ignore cache mismatches. TensorRT 是否忽略 cache 不匹配。</param>
    public void SetTimingCache(TensorRtTimingCache cache, bool ignoreMismatch = false)
    {
        if (cache == null)
        {
            throw new ArgumentNullException(nameof(cache));
        }

        if (cache.Line != Line)
        {
            throw new ArgumentException("Timing cache must belong to the same TensorRT API line as the builder config.");
        }

        NativeBridgeApi.SetTimingCache(Line, _handle, cache.Handle, ignoreMismatch);
    }

    /// <summary>
    /// Releases the TensorRT builder-configuration handle.
    /// 释放 TensorRT builder 配置句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        TensorRtProgressMonitor? monitor = _progressMonitorKeepAlive;
        _disposed = true;
        if (monitor != null)
        {
            TryClearProgressMonitorForDispose();
        }

        _handle.Dispose();
        GC.KeepAlive(monitor);
        DetachProgressMonitor();
        GC.SuppressFinalize(this);
    }

    private void TryClearProgressMonitorForDispose()
    {
        try
        {
            NativeBridgeApi.ClearBuilderConfigProgressMonitor(Line, _handle);
        }
        catch (BridgeProbeException)
        {
            // Dispose must still release the config handle. Keep the progress monitor alive until
            // the config handle is released so TensorRT never observes a freed borrowed monitor.
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtBuilderConfig));
        }
    }

    private TensorRtProgressMonitor? DetachProgressMonitor()
    {
        TensorRtProgressMonitor? monitor = _progressMonitorKeepAlive;
        if (monitor != null)
        {
            _progressMonitorKeepAlive = null;
            monitor.DetachBorrower();
        }

        return monitor;
    }

    private void ValidateLayer(TensorRtLayer layer)
    {
        if (layer == null)
        {
            throw new ArgumentNullException(nameof(layer));
        }

        if (layer.Line != Line)
        {
            throw new ArgumentException("Layer must belong to the same TensorRT API line as the builder config.");
        }
    }
}
