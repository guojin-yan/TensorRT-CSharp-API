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
public sealed partial class TensorRtBuilderConfig
{
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

}
