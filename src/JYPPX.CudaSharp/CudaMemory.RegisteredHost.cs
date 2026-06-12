using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    public void CopyFromAsync(CudaRegisteredHostMemory source, int count, CudaStream stream)
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

    public void CopyToAsync(CudaRegisteredHostMemory destination, int count, CudaStream stream)
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
}
