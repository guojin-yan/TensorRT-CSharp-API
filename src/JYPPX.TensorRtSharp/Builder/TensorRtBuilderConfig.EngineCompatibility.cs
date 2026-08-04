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

}
