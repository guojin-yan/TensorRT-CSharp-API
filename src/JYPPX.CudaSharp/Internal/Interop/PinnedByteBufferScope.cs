using System;
using System.Runtime.InteropServices;

namespace JYPPX.CudaSharp.Internal.Interop;

internal sealed class PinnedByteBufferScope : IDisposable
{
    private GCHandle _handle;

    private PinnedByteBufferScope(byte[] buffer, int size, PinnedByteBufferDescriptor descriptor)
    {
        Buffer = buffer;
        Size = (UIntPtr)size;
        Descriptor = descriptor;
        _handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        Pointer = _handle.AddrOfPinnedObject();
    }

    public byte[] Buffer { get; }
    public PinnedByteBufferDescriptor Descriptor { get; }
    public IntPtr Pointer { get; }
    public UIntPtr Size { get; }

    public static PinnedByteBufferScope Pin(byte[] buffer, int size, PinnedByteBufferDescriptor descriptor)
    {
        if (buffer == null)
        {
            throw new ArgumentNullException(descriptor.BufferParameterName);
        }

        if (size < 0 || size > buffer.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(size));
        }

        return new PinnedByteBufferScope(buffer, size, descriptor);
    }

    public void Dispose()
    {
        if (_handle.IsAllocated)
        {
            _handle.Free();
        }
    }
}
