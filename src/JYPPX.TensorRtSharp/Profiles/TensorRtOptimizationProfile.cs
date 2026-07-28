using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Wraps a TensorRT optimization profile used to describe dynamic input ranges during engine building.
/// 封装 TensorRT 优化配置文件，用于在构建引擎时描述动态输入的 min/opt/max 范围。
/// </summary>
public sealed partial class TensorRtOptimizationProfile : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line that owns this profile.
    /// 获取创建该 profile 的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the extra memory target fraction for this profile.
    /// 获取或设置该 profile 允许 TensorRT 为额外 profile 分配的额外内存比例。
    /// </summary>
    /// <remarks>
    /// TensorRT uses this value as a build-time hint. Valid values are vendor-defined and may vary by TensorRT line.
    /// TensorRT 会把该值作为构建期提示；可接受范围由 TensorRT 版本决定。
    /// </remarks>
    public float ExtraMemoryTarget
    {
        get => NativeBridgeApi.GetOptimizationProfileExtraMemoryTarget(Line, _handle);
        set => NativeBridgeApi.SetOptimizationProfileExtraMemoryTarget(Line, _handle, value);
    }

    /// <summary>
    /// Gets a value indicating whether TensorRT considers this profile complete and internally valid.
    /// 获取 TensorRT 是否认为该 profile 已完整配置且内部状态有效。
    /// </summary>
    public bool IsValid => NativeBridgeApi.IsOptimizationProfileValid(Line, _handle);

    /// <summary>
    /// Sets one min/opt/max dimension selector for a dynamic input tensor.
    /// 设置动态输入张量在某一个 min/opt/max selector 下的维度。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <param name="selector">Profile selector to set. 要设置的 profile selector。</param>
    /// <param name="dims">Tensor dimensions for the selector. 该 selector 对应的张量维度。</param>
    public void SetShape(string inputName, TensorRtOptimizationProfileSelector selector, TensorRtDims dims)
    {
        NativeBridgeApi.SetOptimizationProfileShape(Line, _handle, inputName, selector, dims);
    }

    /// <summary>
    /// Sets the complete min/opt/max dimension range for a dynamic input tensor.
    /// 一次性设置动态输入张量完整的 min/opt/max 维度范围。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <param name="min">Minimum supported shape. 最小支持形状。</param>
    /// <param name="opt">Optimization target shape. 优化目标形状。</param>
    /// <param name="max">Maximum supported shape. 最大支持形状。</param>
    public void SetShape(string inputName, TensorRtDims min, TensorRtDims opt, TensorRtDims max)
    {
        SetShape(inputName, TensorRtOptimizationProfileSelector.Min, min);
        SetShape(inputName, TensorRtOptimizationProfileSelector.Opt, opt);
        SetShape(inputName, TensorRtOptimizationProfileSelector.Max, max);
    }

    /// <summary>
    /// Gets one min/opt/max dimension selector from this optimization profile.
    /// 从该优化 profile 中读取某一个 min/opt/max selector 的维度。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <param name="selector">Profile selector to query. 要查询的 profile selector。</param>
    /// <returns>The TensorRT dimensions stored for the selector. 该 selector 保存的 TensorRT 维度。</returns>
    public TensorRtDims GetShape(string inputName, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetOptimizationProfileShape(Line, _handle, inputName, selector);
    }

    /// <summary>
    /// Gets one TensorRT 11 min/opt/max dimension selector with 64-bit extents.
    /// 以 64 位 extent 获取 TensorRT 11 某个 min/opt/max selector 的维度。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <param name="selector">Profile selector to query. 要查询的 profile selector。</param>
    /// <returns>The TensorRT 11 dimensions stored for the selector. 该 selector 保存的 TensorRT 11 维度。</returns>
    public TensorRtDims64 GetShape64(string inputName, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetOptimizationProfileShape64(Line, _handle, inputName, selector);
    }

    /// <summary>
    /// Gets one dimension extent from a TensorRT 11 profile selector as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 profile selector 的单个维度 extent。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <param name="selector">Profile selector to query. 要查询的 profile selector。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The profile dimension extent reported by TensorRT. TensorRT 报告的 profile 维度 extent。</returns>
    public long GetShapeDimensionExtent64(string inputName, TensorRtOptimizationProfileSelector selector, int dimensionIndex)
    {
        return NativeBridgeApi.GetOptimizationProfileShapeDimensionExtent64(Line, _handle, inputName, selector, dimensionIndex);
    }

    /// <summary>
    /// Gets the complete min/opt/max dimension range for an input tensor.
    /// 获取输入张量完整的 min/opt/max 维度范围。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <returns>The complete profile shape range. 完整的 profile 形状范围。</returns>
    public TensorRtOptimizationProfileShapeRange GetShapeRange(string inputName)
    {
        return new TensorRtOptimizationProfileShapeRange(
            GetShape(inputName, TensorRtOptimizationProfileSelector.Min),
            GetShape(inputName, TensorRtOptimizationProfileSelector.Opt),
            GetShape(inputName, TensorRtOptimizationProfileSelector.Max));
    }

    /// <summary>
    /// Gets the complete TensorRT 11 min/opt/max dimension range using 64-bit extents.
    /// 使用 64 位 extent 获取 TensorRT 11 完整的 min/opt/max 维度范围。
    /// </summary>
    /// <param name="inputName">Network input tensor name. 网络输入张量名称。</param>
    /// <returns>The complete 64-bit profile shape range. 完整的 64 位 profile shape 范围。</returns>
    public TensorRtOptimizationProfileShapeRange64 GetShapeRange64(string inputName)
    {
        return new TensorRtOptimizationProfileShapeRange64(
            GetShape64(inputName, TensorRtOptimizationProfileSelector.Min),
            GetShape64(inputName, TensorRtOptimizationProfileSelector.Opt),
            GetShape64(inputName, TensorRtOptimizationProfileSelector.Max));
    }

    /// <summary>
    /// Sets one min/opt/max value selector for a TensorRT shape tensor input.
    /// 设置 TensorRT shape tensor 输入在某一个 min/opt/max selector 下的取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to set. 要设置的 profile selector。</param>
    /// <param name="values">Integer shape values for the selector. 该 selector 对应的整数 shape 值。</param>
    public void SetShapeValues(string inputName, TensorRtOptimizationProfileSelector selector, params int[] values)
    {
        SetShapeValues(inputName, selector, (IReadOnlyList<int>)values);
    }

    /// <summary>
    /// Sets one min/opt/max value selector for a TensorRT shape tensor input.
    /// 设置 TensorRT shape tensor 输入在某一个 min/opt/max selector 下的取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to set. 要设置的 profile selector。</param>
    /// <param name="values">Integer shape values for the selector. 该 selector 对应的整数 shape 值。</param>
    public void SetShapeValues(string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<int> values)
    {
        NativeBridgeApi.SetOptimizationProfileShapeValues(Line, _handle, inputName, selector, values);
    }

    /// <summary>
    /// Sets the complete min/opt/max value range for a TensorRT shape tensor input.
    /// 一次性设置 TensorRT shape tensor 输入完整的 min/opt/max 取值范围。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <param name="min">Minimum values. 最小取值。</param>
    /// <param name="opt">Optimization target values. 优化目标取值。</param>
    /// <param name="max">Maximum values. 最大取值。</param>
    public void SetShapeValues(string inputName, IReadOnlyList<int> min, IReadOnlyList<int> opt, IReadOnlyList<int> max)
    {
        SetShapeValues(inputName, TensorRtOptimizationProfileSelector.Min, min);
        SetShapeValues(inputName, TensorRtOptimizationProfileSelector.Opt, opt);
        SetShapeValues(inputName, TensorRtOptimizationProfileSelector.Max, max);
    }

    /// <summary>
    /// Gets the number of integer values configured for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入已配置的整数取值数量。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <returns>The number of configured shape values, or a vendor-defined negative value when not configured. 已配置取值数量；未配置时可能返回 TensorRT 定义的负数。</returns>
    public int GetShapeValueCount(string inputName)
    {
        return NativeBridgeApi.GetOptimizationProfileShapeValueCount(Line, _handle, inputName);
    }

    /// <summary>
    /// Gets one min/opt/max value selector for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入在某一个 min/opt/max selector 下的取值。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <param name="selector">Profile selector to query. 要查询的 profile selector。</param>
    /// <returns>The configured shape values. 已配置的 shape 取值。</returns>
    public IReadOnlyList<int> GetShapeValues(string inputName, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetOptimizationProfileShapeValues(Line, _handle, inputName, selector);
    }

    /// <summary>
    /// Gets the complete min/opt/max value range for a TensorRT shape tensor input.
    /// 获取 TensorRT shape tensor 输入完整的 min/opt/max 取值范围。
    /// </summary>
    /// <param name="inputName">Shape tensor input name. shape tensor 输入名称。</param>
    /// <returns>The configured shape-value range. 已配置的 shape-value 范围。</returns>
    public TensorRtOptimizationProfileShapeValueRange GetShapeValueRange(string inputName)
    {
        return new TensorRtOptimizationProfileShapeValueRange(
            GetShapeValues(inputName, TensorRtOptimizationProfileSelector.Min),
            GetShapeValues(inputName, TensorRtOptimizationProfileSelector.Opt),
            GetShapeValues(inputName, TensorRtOptimizationProfileSelector.Max));
    }

    /// <summary>
    /// Releases the native optimization profile handle.
    /// 释放原生 optimization profile 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
