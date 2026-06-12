using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets the size of streamable weights reported by TensorRT.
    /// 获取 TensorRT 报告的可流式加载权重大小。
    /// </summary>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public long StreamableWeightsSizeInBytes => NativeBridgeApi.GetEngineStreamableWeightsSize(Line, _handle);

    /// <summary>
    /// Gets the currently configured TensorRT weight-streaming budget.
    /// 获取当前配置的 TensorRT 权重流式加载预算。
    /// </summary>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public long WeightStreamingBudgetV2InBytes => NativeBridgeApi.GetEngineWeightStreamingBudgetV2(Line, _handle);

    /// <summary>
    /// Gets TensorRT's automatically selected weight-streaming budget.
    /// 获取 TensorRT 自动选择的权重流式加载预算。
    /// </summary>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public long WeightStreamingAutomaticBudgetInBytes => NativeBridgeApi.GetEngineWeightStreamingAutomaticBudget(Line, _handle);

    /// <summary>
    /// Gets the scratch-memory requirement for TensorRT weight streaming.
    /// 获取 TensorRT 权重流式加载所需的临时显存大小。
    /// </summary>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public long WeightStreamingScratchMemorySizeInBytes => NativeBridgeApi.GetEngineWeightStreamingScratchMemorySize(Line, _handle);

    /// <summary>
    /// Gets the hardware-compatibility level recorded on this TensorRT engine.
    /// 获取当前 TensorRT engine 记录的硬件兼容级别。
    /// </summary>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public TensorRtHardwareCompatibilityLevel EngineHardwareCompatibilityLevel => NativeBridgeApi.GetEngineHardwareCompatibilityLevel(Line, _handle);

    /// <summary>
    /// Sets the TensorRT weight-streaming budget.
    /// 设置 TensorRT 权重流式加载预算。
    /// </summary>
    /// <param name="budgetBytes">Budget in bytes. / 预算字节数。</param>
    /// <returns><c>true</c> when TensorRT accepted the budget. / TensorRT 接受该预算时返回 <c>true</c>。</returns>
    /// <remarks>
    /// Supported by this bridge for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public bool SetWeightStreamingBudgetV2(long budgetBytes)
    {
        return NativeBridgeApi.SetEngineWeightStreamingBudgetV2(Line, _handle, budgetBytes);
    }

    /// <summary>
    /// Gets a TensorRT engine statistic.
    /// 获取 TensorRT engine 统计项。
    /// </summary>
    /// <param name="stat">The statistic to query. / 要查询的统计项。</param>
    /// <returns>The statistic value returned by TensorRT. / TensorRT 返回的统计值。</returns>
    /// <remarks>
    /// This specific statistic API is currently exposed only for TensorRT 11 because TensorRT 10 headers do not expose <c>ICudaEngine::getEngineStat</c>.
    /// 该统计接口当前仅面向 TensorRT 11 暴露，因为 TensorRT 10 头文件没有公开 <c>ICudaEngine::getEngineStat</c>。
    /// </remarks>
    public long GetEngineStat(TensorRtEngineStat stat)
    {
        return NativeBridgeApi.GetEngineStat(Line, _handle, stat);
    }
}
