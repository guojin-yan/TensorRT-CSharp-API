using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the Pooling Type value.
    /// 获取 Pooling Type 值。
    /// </summary>
    public TensorRtPoolingType GetPoolingType()
    {
        return NativeBridgeApi.GetPoolingType(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Type value.
    /// 设置 Pooling Type 值。
    /// </summary>
    public void SetPoolingType(TensorRtPoolingType poolingType)
    {
        NativeBridgeApi.SetPoolingType(Line, _handle, poolingType);
    }

    /// <summary>
    /// Sets the Pooling Window Size value.
    /// 设置 Pooling Window Size 值。
    /// </summary>
    public void SetPoolingWindowSize(TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeBridgeApi.SetPoolingWindowSize(Line, _handle, windowSize);
    }

    /// <summary>
    /// Gets the Pooling Window Size value.
    /// 获取 Pooling Window Size 值。
    /// </summary>
    public TensorRtDims GetPoolingWindowSize()
    {
        return NativeBridgeApi.GetPoolingWindowSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Stride value.
    /// 设置 Pooling Stride 值。
    /// </summary>
    public void SetPoolingStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetPoolingStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the Pooling Stride value.
    /// 获取 Pooling Stride 值。
    /// </summary>
    public TensorRtDims GetPoolingStride()
    {
        return NativeBridgeApi.GetPoolingStride(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Padding value.
    /// 设置 Pooling Padding 值。
    /// </summary>
    public void SetPoolingPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Padding value.
    /// 获取 Pooling Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPadding()
    {
        return NativeBridgeApi.GetPoolingPadding(Line, _handle);
    }

    /// <summary>
    /// Gets the Pooling Blend Factor value.
    /// 获取 Pooling Blend Factor 值。
    /// </summary>
    public double GetPoolingBlendFactor()
    {
        return NativeBridgeApi.GetPoolingBlendFactor(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Blend Factor value.
    /// 设置 Pooling Blend Factor 值。
    /// </summary>
    public void SetPoolingBlendFactor(double blendFactor)
    {
        NativeBridgeApi.SetPoolingBlendFactor(Line, _handle, blendFactor);
    }

    /// <summary>
    /// Gets the Pooling Average Count Excludes Padding value.
    /// 获取 Pooling Average Count Excludes Padding 值。
    /// </summary>
    public bool GetPoolingAverageCountExcludesPadding()
    {
        return NativeBridgeApi.GetPoolingAverageCountExcludesPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Average Count Excludes Padding value.
    /// 设置 Pooling Average Count Excludes Padding 值。
    /// </summary>
    public void SetPoolingAverageCountExcludesPadding(bool excludesPadding)
    {
        NativeBridgeApi.SetPoolingAverageCountExcludesPadding(Line, _handle, excludesPadding);
    }

    /// <summary>
    /// Gets the Pooling Pre Padding value.
    /// 获取 Pooling Pre Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPrePadding()
    {
        return NativeBridgeApi.GetPoolingPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Pre Padding value.
    /// 设置 Pooling Pre Padding 值。
    /// </summary>
    public void SetPoolingPrePadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Post Padding value.
    /// 获取 Pooling Post Padding 值。
    /// </summary>
    public TensorRtDims GetPoolingPostPadding()
    {
        return NativeBridgeApi.GetPoolingPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Post Padding value.
    /// 设置 Pooling Post Padding 值。
    /// </summary>
    public void SetPoolingPostPadding(TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeBridgeApi.SetPoolingPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Pooling Padding Mode value.
    /// 获取 Pooling Padding Mode 值。
    /// </summary>
    public TensorRtPaddingMode GetPoolingPaddingMode()
    {
        return NativeBridgeApi.GetPoolingPaddingMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Pooling Padding Mode value.
    /// 设置 Pooling Padding Mode 值。
    /// </summary>
    public void SetPoolingPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetPoolingPaddingMode(Line, _handle, paddingMode);
    }

}
