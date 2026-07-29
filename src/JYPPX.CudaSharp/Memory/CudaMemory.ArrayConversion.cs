using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Materializes a byte array from device memory.
    /// 从设备内存中生成字节数组。
    /// </summary>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <returns>A managed byte array copy. 托管字节数组副本。</returns>
    public byte[] ToArray(int count)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        byte[] data = new byte[count];
        CopyTo(data);
        return data;
    }

    /// <summary>
    /// Materializes a float array from device memory.
    /// 从设备内存中生成浮点数组。
    /// </summary>
    /// <param name="elementCount">The number of single-precision elements to copy. 要复制的单精度元素数量。</param>
    /// <returns>A managed float array copy. 托管浮点数组副本。</returns>
    public float[] ToSingleArray(int elementCount)
    {
        if (elementCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        int byteCount = checked(elementCount * sizeof(float));
        ValidateCount(byteCount, nameof(elementCount));
        float[] data = new float[elementCount];
        CopyTo(data);
        return data;
    }

}
