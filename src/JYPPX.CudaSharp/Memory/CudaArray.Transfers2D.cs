using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaArray
{
    /// <summary>
    /// Copies a 2D managed byte buffer into this CUDA array. 将二维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom2D(byte[] source, int sourcePitch, int widthBytes, int height)
    {
        CopyFrom2D(source, sourcePitch, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D managed byte buffer into this CUDA array at the specified destination offset. 在指定目标偏移处复制二维托管字节缓冲区。
    /// </summary>
    public void CopyFrom2D(byte[] source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height)
    {
        Validate2DRegion(sourcePitch, widthBytes, height, nameof(sourcePitch));
        NativeCudaApi.Copy2DToArray(_handle, destinationXBytes, destinationY, source, sourcePitch, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into a managed byte buffer. 将当前 CUDA array 的二维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo2D(byte[] destination, int destinationPitch, int widthBytes, int height)
    {
        CopyTo2D(destination, destinationPitch, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into a managed byte buffer using an explicit source offset. 使用显式源偏移复制二维区域到托管字节缓冲区。
    /// </summary>
    public void CopyTo2D(byte[] destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        Validate2DRegion(destinationPitch, widthBytes, height, nameof(destinationPitch));
        NativeCudaApi.Copy2DFromArray(destination, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into another CUDA array. 将当前 CUDA array 的二维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo2D(CudaArray destination, int widthBytes, int height)
    {
        CopyTo2D(destination, 0, 0, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region between CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在 CUDA array 之间复制二维区域。
    /// </summary>
    public void CopyTo2D(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate2DExtent(widthBytes, height);
        NativeCudaApi.Copy2DArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D pinned host buffer into this CUDA array asynchronously. 异步把二维 pinned host 缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int widthBytes, int height, CudaStream stream)
    {
        CopyFrom2DAsync(source, sourcePitch, 0, 0, widthBytes, height, stream);
    }

    /// <summary>
    /// Copies a 2D pinned host buffer into this CUDA array asynchronously at the specified destination offset. 在指定目标偏移处异步复制二维 pinned host 缓冲区。
    /// </summary>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(source, sourcePitch, widthBytes, height, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy2DToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, sourcePitch, widthBytes, height, stream.Handle);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into pinned host memory asynchronously. 异步把当前 CUDA array 的二维区域复制到 pinned host memory。
    /// </summary>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int widthBytes, int height, CudaStream stream)
    {
        CopyTo2DAsync(destination, destinationPitch, 0, 0, widthBytes, height, stream);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into pinned host memory asynchronously with an explicit source offset. 使用显式源偏移异步复制二维区域到 pinned host memory。
    /// </summary>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(destination, destinationPitch, widthBytes, height, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy2DFromArrayAsync(destination.Handle, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height, stream.Handle);
    }

}
