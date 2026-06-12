using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Keeps TensorRT refit weights pinned while the native refitter may need to read them.
/// 在原生 refitter 可能读取权重期间，保持 TensorRT refit 权重处于 pinned 状态。
/// </summary>
public sealed class TensorRtRefitWeightsBuffer : IDisposable
{
    private TensorRtWeights.PinnedScope? _pinned;

    /// <summary>
    /// Pins a TensorRT weights object for refitter usage.
    /// 固定一个 TensorRT 权重对象，以供 refitter 使用。
    /// </summary>
    /// <param name="weights">The non-empty weights to pin. 要固定的非空权重。</param>
    public TensorRtRefitWeightsBuffer(TensorRtWeights weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (weights.IsEmpty)
        {
            throw new ArgumentException("Refit weights must not be empty.", nameof(weights));
        }

        DataType = weights.DataType;
        ElementCount = weights.ElementCount;
        _pinned = weights.Pin();
    }

    /// <summary>
    /// Gets the TensorRT data type represented by the pinned weights.
    /// 获取 pinned 权重表示的 TensorRT 数据类型。
    /// </summary>
    public TensorRtDataType DataType { get; }

    /// <summary>
    /// Gets the number of TensorRT weight elements.
    /// 获取 TensorRT 权重元素数量。
    /// </summary>
    public int ElementCount { get; }

    internal IntPtr Pointer
    {
        get
        {
            ThrowIfDisposed();
            return _pinned!.Pointer;
        }
    }

    /// <summary>
    /// Creates a pinned refit buffer from single-precision floating-point weights.
    /// 从单精度浮点权重创建 pinned refit buffer。
    /// </summary>
    /// <param name="values">The weight values. 权重值。</param>
    /// <returns>A pinned refit buffer. pinned refit buffer。</returns>
    public static TensorRtRefitWeightsBuffer FromSingleArray(float[] values)
    {
        return new TensorRtRefitWeightsBuffer(TensorRtWeights.FromSingleArray(values));
    }

    /// <summary>
    /// Creates a pinned refit buffer from 32-bit integer weights.
    /// 从 32 位整数权重创建 pinned refit buffer。
    /// </summary>
    /// <param name="values">The weight values. 权重值。</param>
    /// <returns>A pinned refit buffer. pinned refit buffer。</returns>
    public static TensorRtRefitWeightsBuffer FromInt32Array(int[] values)
    {
        return new TensorRtRefitWeightsBuffer(TensorRtWeights.FromInt32Array(values));
    }

    /// <summary>
    /// Creates a pinned refit buffer from byte-backed INT8 or UINT8 weights.
    /// 从 byte 承载的 INT8 或 UINT8 权重创建 pinned refit buffer。
    /// </summary>
    /// <param name="values">The weight values. 权重值。</param>
    /// <param name="dataType">The TensorRT byte-backed data type. TensorRT byte 承载数据类型。</param>
    /// <returns>A pinned refit buffer. pinned refit buffer。</returns>
    public static TensorRtRefitWeightsBuffer FromByteArray(byte[] values, TensorRtDataType dataType = TensorRtDataType.Int8)
    {
        return new TensorRtRefitWeightsBuffer(TensorRtWeights.FromByteArray(values, dataType));
    }

    /// <summary>
    /// Releases the pinned weights.
    /// 释放 pinned 权重。
    /// </summary>
    public void Dispose()
    {
        _pinned?.Dispose();
        _pinned = null;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_pinned == null)
        {
            throw new ObjectDisposedException(nameof(TensorRtRefitWeightsBuffer));
        }
    }
}
