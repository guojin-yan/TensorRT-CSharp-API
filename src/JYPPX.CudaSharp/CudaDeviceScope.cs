using System;

namespace JYPPX.CudaSharp;

public sealed class CudaDeviceScope : IDisposable
{
    private readonly int _previousDevice;
    private bool _disposed;

    internal CudaDeviceScope(int ordinal)
    {
        _previousDevice = CudaDevice.Current;
        CudaDevice.SetCurrent(ordinal);
        CurrentDevice = ordinal;
    }

    public int CurrentDevice { get; }

    public int PreviousDevice => _previousDevice;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        CudaDevice.SetCurrent(_previousDevice);
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
