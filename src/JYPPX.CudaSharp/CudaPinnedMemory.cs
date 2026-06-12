using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around CUDA pinned host memory.
/// </summary>
public sealed class CudaPinnedMemory : IDisposable
{
    private readonly SafeCudaPinnedMemoryHandle _handle;

    public CudaPinnedMemory(int sizeInBytes)
        : this(sizeInBytes, CudaPinnedMemoryAllocationFlags.Default)
    {
    }

    public CudaPinnedMemory(int sizeInBytes, CudaPinnedMemoryAllocationFlags flags)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = flags == CudaPinnedMemoryAllocationFlags.Default
            ? NativeCudaApi.AllocatePinnedMemory(sizeInBytes)
            : NativeCudaApi.AllocatePinnedMemory(sizeInBytes, flags);
        SizeInBytes = checked((int)NativeCudaApi.GetPinnedMemorySize(_handle));
    }

    internal SafeCudaPinnedMemoryHandle Handle => _handle;

    internal IntPtr Pointer => NativeCudaApi.GetPinnedMemoryPointer(_handle);

    public int SizeInBytes { get; }

    public CudaPinnedMemoryAllocationFlags Flags => NativeCudaApi.GetPinnedMemoryFlags(_handle);

    public bool IsMapped => (Flags & CudaPinnedMemoryAllocationFlags.Mapped) != 0;

    public ulong MappedDevicePointerAddress => unchecked((ulong)NativeCudaApi.GetPinnedMemoryMappedDevicePointer(_handle).ToInt64());

    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFrom(source, source.Length);
    }

    public void CopyFrom(float[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int byteCount = checked(source.Length * sizeof(float));
        byte[] bytes = new byte[byteCount];
        Buffer.BlockCopy(source, 0, bytes, 0, byteCount);
        CopyFrom(bytes, byteCount);
    }

    public void CopyFrom(byte[] source, int count)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        ValidateCount(count, source.Length, nameof(count));
        Marshal.Copy(source, 0, Pointer, count);
    }

    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, destination.Length);
    }

    public void CopyTo(float[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int byteCount = checked(destination.Length * sizeof(float));
        byte[] bytes = new byte[byteCount];
        CopyTo(bytes, byteCount);
        Buffer.BlockCopy(bytes, 0, destination, 0, byteCount);
    }

    public void CopyTo(byte[] destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, destination.Length, nameof(count));
        Marshal.Copy(Pointer, destination, 0, count);
    }

    public byte[] ToArray(int count)
    {
        ValidateCount(count, count, nameof(count));
        byte[] data = new byte[count];
        CopyTo(data, count);
        return data;
    }

    public float[] ToSingleArray(int elementCount)
    {
        if (elementCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        int byteCount = checked(elementCount * sizeof(float));
        ValidateCount(byteCount, byteCount, nameof(elementCount));
        float[] data = new float[elementCount];
        CopyTo(data);
        return data;
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCount(int count, int managedBufferLength, string parameterName)
    {
        if (count < 0 || count > SizeInBytes || count > managedBufferLength)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }
}
