using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaArray
{
    /// <summary>
    /// Copies a 3D managed byte buffer into this CUDA array. 将三维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth)
    {
        CopyFrom3D(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D managed byte buffer into this CUDA array at the specified destination offset. 在指定目标偏移处把三维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(sourcePitch));
        NativeCudaApi.Copy3DToArray(_handle, destinationXBytes, destinationY, destinationZ, source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into a managed byte buffer. 将当前 CUDA array 的三维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into a managed byte buffer using an explicit source offset. 使用显式源偏移将三维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destinationPitch));
        NativeCudaApi.Copy3DFromArray(destination, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into another CUDA array. 将当前 CUDA array 的三维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo3D(CudaArray destination, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region between CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在两个 CUDA array 之间复制三维区域。
    /// </summary>
    public void CopyTo3D(CudaArray destination, int destinationXBytes, int destinationY, int destinationZ, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate3DExtent(widthBytes, height, depth);
        NativeCudaApi.Copy3DArrayToArray(destination._handle, destinationXBytes, destinationY, destinationZ, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D pinned host buffer into this CUDA array asynchronously. 异步把三维 pinned host 缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyFrom3DAsync(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D pinned host buffer into this CUDA array asynchronously at the specified destination offset. 在指定目标偏移处异步复制三维 pinned host 缓冲区。
    /// </summary>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy3DToArrayAsync(_handle, destinationXBytes, destinationY, destinationZ, source.Handle, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into pinned host memory asynchronously. 异步把当前 CUDA array 的三维区域复制到 pinned host memory。
    /// </summary>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into pinned host memory asynchronously using an explicit source offset. 使用显式源偏移异步复制三维区域到 pinned host memory。
    /// </summary>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy3DFromArrayAsync(destination.Handle, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into another CUDA array asynchronously. 异步把当前 CUDA array 的三维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo3DAsync(CudaArray destination, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D region between CUDA arrays asynchronously using explicit source and destination offsets. 使用显式源/目标偏移异步复制三维区域。
    /// </summary>
    public void CopyTo3DAsync(CudaArray destination, int destinationXBytes, int destinationY, int destinationZ, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateStream(stream);
        Validate3DExtent(widthBytes, height, depth);
        NativeCudaApi.Copy3DArrayToArrayAsync(destination._handle, destinationXBytes, destinationY, destinationZ, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth, stream.Handle);
    }

}
