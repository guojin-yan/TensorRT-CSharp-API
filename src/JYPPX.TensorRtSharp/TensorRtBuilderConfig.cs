using System;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilderConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

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

    public void SetFlag(TensorRtBuilderFlag flag, bool enabled = true)
    {
        NativeBridgeApi.SetBuilderConfigFlag(Line, _handle, flag, enabled);
    }

    public void ClearFlag(TensorRtBuilderFlag flag)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            NativeBridgeApi.ClearBuilderConfigFlag(Line, _handle, flag);
            return;
        }

        SetFlag(flag, false);
    }

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
    /// <param name="platform">The TensorRT runtime platform. TensorRT runtime platform。</param>
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

    public void SetLayerDeviceType(TensorRtLayer layer, TensorRtDeviceType deviceType)
    {
        ValidateLayer(layer);
        NativeBridgeApi.SetLayerDeviceType(Line, _handle, layer.Handle, deviceType);
    }

    public TensorRtDeviceType GetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.GetLayerDeviceType(Line, _handle, layer.Handle);
    }

    public bool IsLayerDeviceTypeSet(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        return NativeBridgeApi.IsLayerDeviceTypeSet(Line, _handle, layer.Handle);
    }

    public void ResetLayerDeviceType(TensorRtLayer layer)
    {
        ValidateLayer(layer);
        NativeBridgeApi.ResetLayerDeviceType(Line, _handle, layer.Handle);
    }

    public void SetMemoryPoolLimit(TensorRtMemoryPoolType pool, ulong bytes)
    {
        NativeBridgeApi.SetMemoryPoolLimit(Line, _handle, pool, bytes);
    }

    public ulong GetMemoryPoolLimit(TensorRtMemoryPoolType pool)
    {
        return NativeBridgeApi.GetMemoryPoolLimit(Line, _handle, pool);
    }

    public void SetOptimizationLevel(int level)
    {
        NativeBridgeApi.SetBuilderOptimizationLevel(Line, _handle, level);
    }

    public int GetOptimizationLevel()
    {
        return NativeBridgeApi.GetBuilderOptimizationLevel(Line, _handle);
    }

    public void SetProfilingVerbosity(TensorRtProfilingVerbosity verbosity)
    {
        NativeBridgeApi.SetProfilingVerbosity(Line, _handle, verbosity);
    }

    public TensorRtProfilingVerbosity GetProfilingVerbosity()
    {
        return NativeBridgeApi.GetProfilingVerbosity(Line, _handle);
    }

    public void SetMaxAuxStreams(int maxStreams)
    {
        NativeBridgeApi.SetMaxAuxStreams(Line, _handle, maxStreams);
    }

    public int GetMaxAuxStreams()
    {
        return NativeBridgeApi.GetMaxAuxStreams(Line, _handle);
    }

    public void SetAverageTimingIterations(int iterations)
    {
        NativeBridgeApi.SetAverageTimingIterations(Line, _handle, iterations);
    }

    public int GetAverageTimingIterations()
    {
        return NativeBridgeApi.GetAverageTimingIterations(Line, _handle);
    }

    public void SetTacticSources(TensorRtTacticSources sources)
    {
        NativeBridgeApi.SetTacticSources(Line, _handle, sources);
    }

    public TensorRtTacticSources GetTacticSources()
    {
        return NativeBridgeApi.GetTacticSources(Line, _handle);
    }

    public TensorRtTimingCache CreateTimingCache(byte[]? serializedCache = null)
    {
        return new TensorRtTimingCache(Line, NativeBridgeApi.CreateTimingCache(Line, _handle, serializedCache));
    }

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

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
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
