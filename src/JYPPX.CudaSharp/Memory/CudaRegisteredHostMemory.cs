using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Registers a managed byte buffer as CUDA pinned host memory for asynchronous transfers.
/// 将托管字节缓冲区注册为 CUDA pinned host memory 以支持异步传输。
/// </summary>
public sealed class CudaRegisteredHostMemory : IDisposable
{
    private readonly SafeCudaPinnedMemoryHandle _handle;
    private GCHandle _pinnedBuffer;
    private bool _disposed;

    /// <summary>
    /// Registers a managed buffer with default CUDA host-registration flags.
    /// 使用默认 CUDA host registration 标志注册托管缓冲区。
    /// </summary>
    /// <param name="buffer">The managed buffer to register. 要注册的托管缓冲区。</param>
    public CudaRegisteredHostMemory(byte[] buffer)
        : this(buffer, CudaHostRegistrationFlags.Default)
    {
    }

    /// <summary>
    /// Registers a managed buffer with explicit CUDA host-registration flags.
    /// 使用显式 CUDA host registration 标志注册托管缓冲区。
    /// </summary>
    /// <param name="buffer">The managed buffer to register. 要注册的托管缓冲区。</param>
    /// <param name="flags">The host-registration flags. host registration 标志。</param>
    public CudaRegisteredHostMemory(byte[] buffer, CudaHostRegistrationFlags flags)
    {
        if (buffer == null)
        {
            throw new ArgumentNullException(nameof(buffer));
        }

        if (buffer.Length == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(buffer), "Registered host memory buffer must not be empty.");
        }

        NativeBridgeLoader.EnsureInitialized();
        Buffer = buffer;
        SizeInBytes = buffer.Length;
        Flags = flags;
        _pinnedBuffer = GCHandle.Alloc(buffer, GCHandleType.Pinned);

        try
        {
            _handle = NativeCudaApi.RegisterPinnedMemory(_pinnedBuffer.AddrOfPinnedObject(), buffer.Length, flags);
        }
        catch
        {
            _pinnedBuffer.Free();
            throw;
        }
    }

    /// <summary>
    /// Gets the registered managed buffer.
    /// 获取已注册的托管缓冲区。
    /// </summary>
    public byte[] Buffer { get; }

    /// <summary>
    /// Gets the buffer size in bytes.
    /// 获取缓冲区大小，单位为字节。
    /// </summary>
    public int SizeInBytes { get; }

    /// <summary>
    /// Gets the CUDA host-registration flags.
    /// 获取 CUDA host registration 标志。
    /// </summary>
    public CudaHostRegistrationFlags Flags { get; }

    /// <summary>
    /// Gets whether the registered host buffer is mapped into device address space.
    /// 获取该注册的主机缓冲区是否映射到了设备地址空间。
    /// </summary>
    public bool IsMapped => (Flags & CudaHostRegistrationFlags.Mapped) != 0;

    /// <summary>
    /// Gets the mapped device pointer address when the registration is mapped.
    /// 在该注册已映射时获取对应的设备指针地址。
    /// </summary>
    public ulong MappedDevicePointerAddress
    {
        get
        {
            ThrowIfDisposed();
            return unchecked((ulong)NativeCudaApi.GetPinnedMemoryMappedDevicePointer(_handle).ToInt64());
        }
    }

    internal SafeCudaPinnedMemoryHandle Handle
    {
        get
        {
            ThrowIfDisposed();
            return _handle;
        }
    }

    /// <summary>
    /// Unregisters the host buffer and releases its pinned handle.
    /// 取消注册主机缓冲区并释放其固定句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _handle.Dispose();
        if (_pinnedBuffer.IsAllocated)
        {
            _pinnedBuffer.Free();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaRegisteredHostMemory));
        }
    }
}
