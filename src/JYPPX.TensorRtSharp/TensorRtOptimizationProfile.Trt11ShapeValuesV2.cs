using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOptimizationProfile
{
    /// <summary>
    /// Sets one min/opt/max 64-bit value selector for a TensorRT shape tensor input.
    /// 为 TensorRT shape tensor 输入设置某个 min/opt/max selector 下的 64 位整数取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to set. 要设置的 profile selector。</param>
    /// <param name="values">64-bit integer shape values for the selector. 该 selector 对应的 64 位整数 shape 值。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the values. 当 TensorRT 接受该取值时返回 <see langword="true"/>。</returns>
    public bool SetShapeValuesV2(string inputName, TensorRtOptimizationProfileSelector selector, params long[] values)
    {
        return SetShapeValuesV2(inputName, selector, (IReadOnlyList<long>)values);
    }

    /// <summary>
    /// Sets one min/opt/max 64-bit value selector for a TensorRT shape tensor input.
    /// 为 TensorRT shape tensor 输入设置某个 min/opt/max selector 下的 64 位整数取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to set. 要设置的 profile selector。</param>
    /// <param name="values">64-bit integer shape values for the selector. 该 selector 对应的 64 位整数 shape 值。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the values. 当 TensorRT 接受该取值时返回 <see langword="true"/>。</returns>
    public bool SetShapeValuesV2(string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<long> values)
    {
        return NativeBridgeApi.SetOptimizationProfileShapeValuesV2(Line, _handle, inputName, selector, values);
    }

    /// <summary>
    /// Sets the complete min/opt/max 64-bit value range for a TensorRT shape tensor input.
    /// 一次性设置 TensorRT shape tensor 输入完整的 min/opt/max 64 位取值范围。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <param name="min">Minimum values. 最小取值。</param>
    /// <param name="opt">Optimization target values. 优化目标取值。</param>
    /// <param name="max">Maximum values. 最大取值。</param>
    public void SetShapeValuesV2(string inputName, IReadOnlyList<long> min, IReadOnlyList<long> opt, IReadOnlyList<long> max)
    {
        SetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Min, min);
        SetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Opt, opt);
        SetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Max, max);
    }

    /// <summary>
    /// Gets the number of 64-bit shape values configured for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入已配置的 64 位 shape 值数量。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <returns>The number of configured values, or a TensorRT-defined negative value when unset. 已配置数量；未配置时可能返回 TensorRT 定义的负数。</returns>
    public int GetShapeValueCountV2(string inputName)
    {
        return NativeBridgeApi.GetOptimizationProfileShapeValueCountV2(Line, _handle, inputName);
    }

    /// <summary>
    /// Gets one min/opt/max 64-bit value selector for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入在某个 min/opt/max selector 下的 64 位取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to query. 要查询的 profile selector。</param>
    /// <returns>The configured 64-bit shape values. 已配置的 64 位 shape 值。</returns>
    public IReadOnlyList<long> GetShapeValuesV2(string inputName, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetOptimizationProfileShapeValuesV2(Line, _handle, inputName, selector);
    }

    /// <summary>
    /// Gets the complete min/opt/max 64-bit value range for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入完整的 min/opt/max 64 位取值范围。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. Shape tensor 输入名称。</param>
    /// <returns>The configured 64-bit shape-value range. 已配置的 64 位 shape-value 范围。</returns>
    public TensorRtOptimizationProfileShapeValueRangeV2 GetShapeValueRangeV2(string inputName)
    {
        return new TensorRtOptimizationProfileShapeValueRangeV2(
            GetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Min),
            GetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Opt),
            GetShapeValuesV2(inputName, TensorRtOptimizationProfileSelector.Max));
    }
}
