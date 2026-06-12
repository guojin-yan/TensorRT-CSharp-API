using System;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp;

public sealed class TensorRtWeights
{
    private TensorRtWeights(TensorRtDataType dataType, byte[] bytes, int elementCount)
    {
        DataType = dataType;
        Bytes = bytes;
        ElementCount = elementCount;
    }

    public TensorRtDataType DataType { get; }

    public int ElementCount { get; }

    internal byte[] Bytes { get; }

    internal bool IsEmpty => ElementCount == 0;

    public static TensorRtWeights Empty(TensorRtDataType dataType = TensorRtDataType.Float)
    {
        return new TensorRtWeights(dataType, Array.Empty<byte>(), 0);
    }

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
