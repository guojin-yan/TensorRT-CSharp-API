using System;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Immutable TensorRT weights payload for layer creation APIs.
/// 用于 layer 创建 API 的不可变 TensorRT weights 负载。
/// </summary>
public sealed class TensorRtWeights
{
    private TensorRtWeights(TensorRtDataType dataType, byte[] bytes, int elementCount)
    {
        DataType = dataType;
        Bytes = bytes;
        ElementCount = elementCount;
    }

    /// <summary>
    /// Gets the TensorRT data type carried by the weights payload.
    /// 获取当前 weights 负载携带的 TensorRT 数据类型。
    /// </summary>
    public TensorRtDataType DataType { get; }

    /// <summary>
    /// Gets the logical element count.
    /// 获取逻辑元素数量。
    /// </summary>
    public int ElementCount { get; }

    internal byte[] Bytes { get; }

    internal bool IsEmpty => ElementCount == 0;

    /// <summary>
    /// Creates an empty weights payload for the requested TensorRT data type.
    /// 为指定 TensorRT 数据类型创建一个空的 weights 负载。
    /// </summary>
    /// <param name="dataType">The TensorRT data type. TensorRT 数据类型。</param>
    /// <returns>An empty weights payload. 空的 weights 负载。</returns>
    public static TensorRtWeights Empty(TensorRtDataType dataType = TensorRtDataType.Float)
    {
        return new TensorRtWeights(dataType, Array.Empty<byte>(), 0);
    }

    /// <summary>
    /// Creates float weights from a managed single-precision array.
    /// 从托管单精度数组创建 float weights。
    /// </summary>
    /// <param name="values">The source float values. 源浮点值数组。</param>
    /// <returns>A TensorRT weights payload. TensorRT weights 负载。</returns>
    public static TensorRtWeights FromSingleArray(float[] values)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return new TensorRtWeights(TensorRtDataType.Float, bytes, values.Length);
    }

    /// <summary>
    /// Creates INT32 weights from a managed integer array.
    /// 从托管整数数组创建 INT32 weights。
    /// </summary>
    /// <param name="values">The source integer values. 源整数值数组。</param>
    /// <returns>A TensorRT weights payload. TensorRT weights 负载。</returns>
    public static TensorRtWeights FromInt32Array(int[] values)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        byte[] bytes = new byte[checked(values.Length * sizeof(int))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return new TensorRtWeights(TensorRtDataType.Int32, bytes, values.Length);
    }

    /// <summary>
    /// Creates byte-backed weights from a managed byte array.
    /// 从托管字节数组创建 byte-backed weights。
    /// </summary>
    /// <param name="values">The source byte values. 源字节值数组。</param>
    /// <param name="dataType">The TensorRT byte-compatible data type. 兼容字节表示的 TensorRT 数据类型。</param>
    /// <returns>A TensorRT weights payload. TensorRT weights 负载。</returns>
    public static TensorRtWeights FromByteArray(byte[] values, TensorRtDataType dataType = TensorRtDataType.Int8)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (dataType != TensorRtDataType.Int8 && dataType != TensorRtDataType.UInt8)
        {
            throw new ArgumentException("Byte-backed weights currently support Int8 or UInt8 only.", nameof(dataType));
        }

        byte[] bytes = new byte[values.Length];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return new TensorRtWeights(dataType, bytes, values.Length);
    }

    /// <summary>
    /// Creates boolean weights from a managed boolean array.
    /// 从托管布尔数组创建 boolean weights。
    /// </summary>
    /// <param name="values">The source boolean values. 源布尔值数组。</param>
    /// <returns>A TensorRT weights payload. TensorRT weights 负载。</returns>
    public static TensorRtWeights FromBooleanArray(bool[] values)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        byte[] bytes = new byte[values.Length];
        for (int index = 0; index < values.Length; index++)
        {
            bytes[index] = values[index] ? (byte)1 : (byte)0;
        }

        return new TensorRtWeights(TensorRtDataType.Bool, bytes, values.Length);
    }

    internal PinnedScope Pin()
    {
        return new PinnedScope(Bytes);
    }

    internal sealed class PinnedScope : IDisposable
    {
        private GCHandle _handle;

        public PinnedScope(byte[] bytes)
        {
            if (bytes.Length == 0)
            {
                throw new ArgumentException("Weights must not be empty.", nameof(bytes));
            }

            _handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();
        }

        public IntPtr Pointer { get; }

        public void Dispose()
        {
            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
        }
    }
}
