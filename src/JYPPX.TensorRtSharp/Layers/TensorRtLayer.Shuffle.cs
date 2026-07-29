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
    /// Sets the Shuffle Reshape Dimensions value.
    /// 设置 Shuffle Reshape Dimensions 值。
    /// </summary>
    public void SetShuffleReshapeDimensions(TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeBridgeApi.SetShuffleReshapeDimensions(Line, _handle, dims);
    }

    /// <summary>
    /// Gets the Shuffle Reshape Dimensions value.
    /// 获取 Shuffle Reshape Dimensions 值。
    /// </summary>
    public TensorRtDims GetShuffleReshapeDimensions()
    {
        return NativeBridgeApi.GetShuffleReshapeDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle layer permutation applied before reshaping.
    /// 设置 Shuffle 层在 reshape 之前应用的转置排列。
    /// </summary>
    /// <param name="permutation">Permutation dimensions. 转置排列维度。</param>
    public void SetShuffleFirstTranspose(TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeBridgeApi.SetShuffleFirstTranspose(Line, _handle, permutation);
    }

    /// <summary>
    /// Gets the Shuffle layer permutation applied before reshaping.
    /// 获取 Shuffle 层在 reshape 之前应用的转置排列。
    /// </summary>
    /// <returns>The first transpose permutation. 第一段转置排列。</returns>
    public TensorRtDims GetShuffleFirstTranspose()
    {
        return NativeBridgeApi.GetShuffleFirstTranspose(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle layer permutation applied after reshaping.
    /// 设置 Shuffle 层在 reshape 之后应用的转置排列。
    /// </summary>
    /// <param name="permutation">Permutation dimensions. 转置排列维度。</param>
    public void SetShuffleSecondTranspose(TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeBridgeApi.SetShuffleSecondTranspose(Line, _handle, permutation);
    }

    /// <summary>
    /// Gets the Shuffle layer permutation applied after reshaping.
    /// 获取 Shuffle 层在 reshape 之后应用的转置排列。
    /// </summary>
    /// <returns>The second transpose permutation. 第二段转置排列。</returns>
    public TensorRtDims GetShuffleSecondTranspose()
    {
        return NativeBridgeApi.GetShuffleSecondTranspose(Line, _handle);
    }

    /// <summary>
    /// Sets the Shuffle Zero Is Placeholder value.
    /// 设置 Shuffle Zero Is Placeholder 值。
    /// </summary>
    public void SetShuffleZeroIsPlaceholder(bool zeroIsPlaceholder)
    {
        NativeBridgeApi.SetShuffleZeroIsPlaceholder(Line, _handle, zeroIsPlaceholder);
    }

    /// <summary>
    /// Gets the Shuffle Zero Is Placeholder value.
    /// 获取 Shuffle Zero Is Placeholder 值。
    /// </summary>
    public bool GetShuffleZeroIsPlaceholder()
    {
        return NativeBridgeApi.GetShuffleZeroIsPlaceholder(Line, _handle);
    }

}
