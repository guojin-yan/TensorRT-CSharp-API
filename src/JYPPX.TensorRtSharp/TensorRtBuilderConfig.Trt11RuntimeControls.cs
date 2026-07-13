using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Sets the complete TensorRT 11 builder flag bitmask.
    /// 设置完整的 TensorRT 11 builder flag 位掩码。
    /// </summary>
    /// <param name="flags">The full builder flag bitmask. / 完整 builder flag 位掩码。</param>
    public void SetFlags(TensorRtBuilderFlags flags)
    {
        NativeBridgeApi.SetBuilderConfigFlags(Line, _handle, flags);
    }

    /// <summary>
    /// Gets the complete TensorRT builder flag bitmask.
    /// 获取完整的 TensorRT builder flag 位掩码。
    /// </summary>
    public TensorRtBuilderFlags GetFlags()
    {
        return NativeBridgeApi.GetBuilderConfigFlags(Line, _handle);
    }

    /// <summary>
    /// Sets the default TensorRT device type for layers.
    /// 设置 TensorRT layer 的默认设备类型。
    /// </summary>
    public void SetDefaultDeviceType(TensorRtDeviceType deviceType)
    {
        NativeBridgeApi.SetBuilderConfigDefaultDeviceType(Line, _handle, deviceType);
    }

    /// <summary>
    /// Gets the default TensorRT device type for layers.
    /// 获取 TensorRT layer 的默认设备类型。
    /// </summary>
    public TensorRtDeviceType GetDefaultDeviceType()
    {
        return NativeBridgeApi.GetBuilderConfigDefaultDeviceType(Line, _handle);
    }

    /// <summary>
    /// Sets the selected DLA core index.
    /// 设置选中的 DLA core 索引。
    /// </summary>
    public void SetDlaCore(int dlaCore)
    {
        NativeBridgeApi.SetBuilderConfigDlaCore(Line, _handle, dlaCore);
    }

    /// <summary>
    /// Gets the selected DLA core index.
    /// 获取选中的 DLA core 索引。
    /// </summary>
    public int GetDlaCore()
    {
        return NativeBridgeApi.GetBuilderConfigDlaCore(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 10/11 tiling optimization level.
    /// 设置 TensorRT 10/11 tiling 优化级别。
    /// </summary>
    /// <param name="level">The tiling optimization level. / tiling 优化级别。</param>
    /// <returns><c>true</c> if TensorRT accepted the setting. / TensorRT 接受该设置时返回 <c>true</c>。</returns>
    public bool SetTilingOptimizationLevel(TensorRtTilingOptimizationLevel level)
    {
        return NativeBridgeApi.SetBuilderConfigTilingOptimizationLevel(Line, _handle, level);
    }

    /// <summary>
    /// Gets the TensorRT 10/11 tiling optimization level.
    /// 获取 TensorRT 10/11 tiling 优化级别。
    /// </summary>
    public TensorRtTilingOptimizationLevel GetTilingOptimizationLevel()
    {
        return NativeBridgeApi.GetBuilderConfigTilingOptimizationLevel(Line, _handle);
    }

    /// <summary>
    /// Sets the L2 byte limit used by TensorRT 10/11 tiling optimization.
    /// 设置 TensorRT 10/11 tiling 优化使用的 L2 字节上限。
    /// </summary>
    /// <param name="bytes">L2 byte limit. / L2 字节上限。</param>
    /// <returns><c>true</c> if TensorRT accepted the setting. / TensorRT 接受该设置时返回 <c>true</c>。</returns>
    public bool SetL2LimitForTiling(long bytes)
    {
        return NativeBridgeApi.SetBuilderConfigL2LimitForTiling(Line, _handle, bytes);
    }

    /// <summary>
    /// Gets the L2 byte limit used by TensorRT 10/11 tiling optimization.
    /// 获取 TensorRT 10/11 tiling 优化使用的 L2 字节上限。
    /// </summary>
    public long GetL2LimitForTiling()
    {
        return NativeBridgeApi.GetBuilderConfigL2LimitForTiling(Line, _handle);
    }

    /// <summary>
    /// Sets the maximum number of TensorRT tactics considered during build.
    /// 设置构建时 TensorRT 最多考虑的 tactic 数量。
    /// </summary>
    public void SetMaxTactics(int maxTactics)
    {
        NativeBridgeApi.SetBuilderConfigMaxTactics(Line, _handle, maxTactics);
    }

    /// <summary>
    /// Gets the maximum number of TensorRT tactics considered during build.
    /// 获取构建时 TensorRT 最多考虑的 tactic 数量。
    /// </summary>
    public int GetMaxTactics()
    {
        return NativeBridgeApi.GetBuilderConfigMaxTactics(Line, _handle);
    }

    /// <summary>
    /// Sets the complete TensorRT 8/10 quantization flag bitmask.
    /// 设置完整的 TensorRT 8/10 quantization flag 位掩码。
    /// </summary>
    /// <param name="flags">The full quantization flag bitmask. / 完整 quantization flag 位掩码。</param>
    public void SetQuantizationFlags(TensorRtQuantizationFlags flags)
    {
        NativeBridgeApi.SetBuilderConfigQuantizationFlags(Line, _handle, flags);
    }

    /// <summary>
    /// Gets the complete TensorRT 8/10 quantization flag bitmask.
    /// 获取完整的 TensorRT 8/10 quantization flag 位掩码。
    /// </summary>
    public TensorRtQuantizationFlags GetQuantizationFlags()
    {
        return NativeBridgeApi.GetBuilderConfigQuantizationFlags(Line, _handle);
    }

    /// <summary>
    /// Enables one TensorRT 8/10 quantization flag.
    /// 启用一个 TensorRT 8/10 quantization flag。
    /// </summary>
    public void SetQuantizationFlag(TensorRtQuantizationFlag flag)
    {
        NativeBridgeApi.SetBuilderConfigQuantizationFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Clears one TensorRT 8/10 quantization flag.
    /// 清除一个 TensorRT 8/10 quantization flag。
    /// </summary>
    public void ClearQuantizationFlag(TensorRtQuantizationFlag flag)
    {
        NativeBridgeApi.ClearBuilderConfigQuantizationFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Returns whether one TensorRT 8/10 quantization flag is enabled.
    /// 返回某个 TensorRT 8/10 quantization flag 当前是否启用。
    /// </summary>
    public bool GetQuantizationFlag(TensorRtQuantizationFlag flag)
    {
        return NativeBridgeApi.GetBuilderConfigQuantizationFlag(Line, _handle, flag);
    }

    /// <summary>
    /// Sets the TensorRT 11 remote auto-tuning configuration text.
    /// 设置 TensorRT 11 remote auto-tuning 配置文本。
    /// </summary>
    /// <param name="configText">Configuration text accepted by TensorRT. / TensorRT 可接受的配置文本。</param>
    /// <returns><c>true</c> if TensorRT accepted the setting. / TensorRT 接受该设置时返回 <c>true</c>。</returns>
    public bool SetRemoteAutoTuningConfig(string configText)
    {
        return NativeBridgeApi.SetBuilderConfigRemoteAutoTuningConfig(Line, _handle, configText);
    }

    /// <summary>
    /// Gets the TensorRT 11 remote auto-tuning configuration text.
    /// 获取 TensorRT 11 remote auto-tuning 配置文本。
    /// </summary>
    public string GetRemoteAutoTuningConfig()
    {
        return NativeBridgeApi.GetBuilderConfigRemoteAutoTuningConfig(Line, _handle);
    }
}
