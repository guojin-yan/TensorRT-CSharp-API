using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaArray
{
    /// <summary>
    /// Copies the full source byte array into this CUDA array. 将整个源字节数组复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFrom(source, source.Length);
    }

    /// <summary>
    /// Copies a fixed number of bytes into this CUDA array. 将固定字节数复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source, int byteCount)
    {
        CopyFrom(source, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes into this CUDA array at the specified 2D offset. 在指定二维偏移处把字节复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source, int destinationXBytes, int destinationY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyToArray(_handle, destinationXBytes, destinationY, source, byteCount);
    }

    /// <summary>
    /// Copies the full contents of this CUDA array into a managed byte array. 将当前 CUDA array 的全部内容复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, destination.Length);
    }

    /// <summary>
    /// Copies a fixed number of bytes from this CUDA array into a managed byte array. 将固定字节数从当前 CUDA array 复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination, int byteCount)
    {
        CopyTo(destination, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes from this CUDA array at the specified 2D offset into a managed byte array. 从指定二维偏移把字节复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination, int sourceXBytes, int sourceY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyFromArray(destination, _handle, sourceXBytes, sourceY, byteCount);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into another CUDA array. 将当前 CUDA array 的字节复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo(CudaArray destination, int byteCount)
    {
        CopyTo(destination, 0, 0, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes between two CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在两个 CUDA array 之间复制字节。
    /// </summary>
    public void CopyTo(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int byteCount)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateByteCount(byteCount);
        NativeCudaApi.CopyArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, byteCount);
    }

    /// <summary>
    /// Copies bytes from pinned host memory into this CUDA array asynchronously. 以异步方式把 pinned host memory 复制到当前 CUDA array。
    /// </summary>
    public void CopyFromAsync(CudaPinnedMemory source, int byteCount, CudaStream stream)
    {
        CopyFromAsync(source, 0, 0, byteCount, stream);
    }

    /// <summary>
    /// Copies bytes from pinned host memory into this CUDA array asynchronously at the specified 2D offset. 在指定二维偏移处异步复制 pinned host memory。
    /// </summary>
    public void CopyFromAsync(CudaPinnedMemory source, int destinationXBytes, int destinationY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(source, byteCount, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.CopyToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, byteCount, stream.Handle);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into pinned host memory asynchronously. 以异步方式把当前 CUDA array 复制到 pinned host memory。
    /// </summary>
    public void CopyToAsync(CudaPinnedMemory destination, int byteCount, CudaStream stream)
    {
        CopyToAsync(destination, 0, 0, byteCount, stream);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into pinned host memory asynchronously using an explicit 2D source offset. 使用显式二维源偏移异步复制到 pinned host memory。
    /// </summary>
    public void CopyToAsync(CudaPinnedMemory destination, int sourceXBytes, int sourceY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(destination, byteCount, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.CopyFromArrayAsync(destination.Handle, _handle, sourceXBytes, sourceY, byteCount, stream.Handle);
    }

}
