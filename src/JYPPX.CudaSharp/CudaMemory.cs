using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a device memory allocation.
/// CUDA 设备内存分配的托管封装。
/// </summary>
public partial class CudaMemory : IDisposable
{
    private readonly SafeCudaMemoryHandle _handle;

    public CudaMemory(int sizeInBytes)
        : this(AllocateDeviceMemory(sizeInBytes))
    {
    }

    internal CudaMemory(SafeCudaMemoryHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        SizeInBytes = checked((int)NativeCudaApi.GetMemorySize(_handle));
    }

    public int SizeInBytes { get; }

    internal SafeCudaMemoryHandle Handle => _handle;

    /// <summary>
    /// Queries CUDA pointer attributes for this device allocation.
    /// 查询当前设备内存分配的 CUDA 指针属性。
    /// </summary>
    /// <returns>Pointer metadata useful for diagnostics and deployment validation. 用于诊断和部署验证的指针元数据。</returns>
    public CudaPointerAttributes GetPointerAttributes()
    {
        NativeCudaPointerAttributes attributes = NativeCudaApi.GetPointerAttributes(_handle);
        return new CudaPointerAttributes(
            (CudaMemoryPointerType)attributes.MemoryType,
            attributes.Device,
            attributes.DevicePointer,
            attributes.HostPointer);
    }

    public static CudaMemory AllocateAsync(int sizeInBytes, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryAsync(sizeInBytes, stream.Handle));
    }

    /// <summary>
    /// Asynchronously allocates device memory from a specific CUDA memory pool.
    /// 从指定 CUDA memory pool 中异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="memoryPool">The CUDA memory pool used for allocation. 用于分配的 CUDA memory pool。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed CUDA memory wrapper. CUDA 设备内存托管封装。</returns>
    public static CudaMemory AllocateFromPoolAsync(int sizeInBytes, CudaMemoryPool memoryPool, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryFromPoolAsync(sizeInBytes, memoryPool.Handle, stream.Handle));
    }

    public void Fill(byte value, int count)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.FillMemory(_handle, value, count);
    }

    /// <summary>
    /// Fills the entire allocation with a byte value.
    /// 使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    public void Fill(byte value)
    {
        Fill(value, SizeInBytes);
    }

    public void FillAsync(byte value, int count, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        NativeCudaApi.FillMemoryAsync(_handle, value, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously fills the entire allocation with a byte value.
    /// 异步使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void FillAsync(byte value, CudaStream stream)
    {
        FillAsync(value, SizeInBytes, stream);
    }

    public void PrefetchAsync(int count, int destinationDevice, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        NativeCudaApi.PrefetchMemoryAsync(_handle, count, destinationDevice, stream.Handle);
    }

    /// <summary>
    /// Asynchronously prefetches the entire allocation to a target CUDA device.
    /// 异步将整个分配预取到目标 CUDA 设备。
    /// </summary>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int destinationDevice, CudaStream stream)
    {
        PrefetchAsync(SizeInBytes, destinationDevice, stream);
    }

    public void Advise(int count, CudaMemoryAdvice advice, int device)
    {
        ValidateCount(count, nameof(count));
        NativeCudaApi.AdviseMemory(_handle, count, (int)advice, device);
    }

    /// <summary>
    /// Applies a CUDA memory advice to the entire allocation.
    /// 对整个分配应用 CUDA memory advice。
    /// </summary>
    /// <param name="advice">The CUDA memory advice. CUDA memory advice。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(CudaMemoryAdvice advice, int device)
    {
        Advise(SizeInBytes, advice, device);
    }

    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        NativeCudaApi.CopyFromHost(_handle, source, source.Length);
    }

    public void CopyFrom(float[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int byteCount = checked(source.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        byte[] bytes = new byte[byteCount];
        Buffer.BlockCopy(source, 0, bytes, 0, byteCount);
        CopyFrom(bytes);
    }

    public void CopyFromAsync(CudaPinnedMemory source, int count, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > source.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyFromPinnedHostAsync(_handle, source.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies the full pinned-host buffer into this device allocation.
    /// 异步将整个 pinned host 缓冲区复制到当前设备内存。
    /// </summary>
    /// <param name="source">The source pinned host buffer. 源 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaPinnedMemory source, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFromAsync(source, Math.Min(source.SizeInBytes, SizeInBytes), stream);
    }

    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (destination.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        NativeCudaApi.CopyToHost(_handle, destination, destination.Length);
    }

    public void CopyTo(float[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int byteCount = checked(destination.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        byte[] bytes = new byte[byteCount];
        CopyTo(bytes);
        Buffer.BlockCopy(bytes, 0, destination, 0, byteCount);
    }

    public void CopyToAsync(CudaPinnedMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyToPinnedHostAsync(_handle, destination.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies this allocation into a full pinned-host buffer.
    /// 异步将当前设备内存复制到整个 pinned host 缓冲区。
    /// </summary>
    /// <param name="destination">The destination pinned host buffer. 目标 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPinnedMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(destination.SizeInBytes, SizeInBytes), stream);
    }

    public void CopyTo(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDeviceToDevice(destination._handle, _handle, count);
    }

    /// <summary>
    /// Copies bytes from this allocation to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    public void CopyTo(CudaMemory destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, Math.Min(SizeInBytes, destination.SizeInBytes));
    }

    public void CopyToAsync(CudaMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDeviceToDeviceAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies bytes to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，异步将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(SizeInBytes, destination.SizeInBytes), stream);
    }

    /// <summary>
    /// Copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToAuto(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDefault(destination._handle, _handle, count);
    }

    /// <summary>
    /// Asynchronously copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 异步将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAutoAsync(CudaMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDefaultAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Copies bytes from this allocation to a destination allocation on another CUDA device.
    /// 将当前分配中的字节复制到另一个 CUDA 设备上的目标分配。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToPeer(CudaMemory destination, int sourceDevice, int destinationDevice, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyPeer(destination._handle, destinationDevice, _handle, sourceDevice, count);
    }

    /// <summary>
    /// Asynchronously copies bytes from this allocation to another CUDA device using a stream.
    /// 使用 stream 将当前分配中的字节异步复制到另一个 CUDA 设备。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序该复制操作的 CUDA stream。</param>
    public void CopyToPeerAsync(CudaMemory destination, int sourceDevice, int destinationDevice, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyPeerAsync(destination._handle, destinationDevice, _handle, sourceDevice, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously frees this allocation on the supplied CUDA stream and invalidates the managed handle.
    /// 在指定 CUDA stream 上异步释放当前分配，并使托管句柄失效。
    /// </summary>
    /// <param name="stream">The CUDA stream that orders the free operation. 用于排序释放操作的 CUDA stream。</param>
    public void FreeAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.FreeMemoryAsync(_handle, stream.Handle);
        _handle.MarkReleased();
    }

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

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCount(int count, string parameterName)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static SafeCudaMemoryHandle AllocateDeviceMemory(int sizeInBytes)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.AllocateMemory(sizeInBytes);
    }
}
