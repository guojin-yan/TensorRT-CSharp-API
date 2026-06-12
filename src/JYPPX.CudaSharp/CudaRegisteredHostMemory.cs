using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Registers a managed byte buffer as CUDA pinned host memory for asynchronous transfers.
/// </summary>
public sealed class CudaRegisteredHostMemory : IDisposable
{
    private readonly SafeCudaPinnedMemoryHandle _handle;
    private GCHandle _pinnedBuffer;
    private bool _disposed;

    public CudaRegisteredHostMemory(byte[] buffer)
        : this(buffer, CudaHostRegistrationFlags.Default)
    {
    }

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

    public byte[] Buffer { get; }

    public int SizeInBytes { get; }

    public CudaHostRegistrationFlags Flags { get; }

    public bool IsMapped => (Flags & CudaHostRegistrationFlags.Mapped) != 0;

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
