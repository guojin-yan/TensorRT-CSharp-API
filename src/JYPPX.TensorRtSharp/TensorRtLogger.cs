using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed class TensorRtLogger : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    public TensorRtLogger(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        _handle = NativeBridgeApi.CreateLogger(line);
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
