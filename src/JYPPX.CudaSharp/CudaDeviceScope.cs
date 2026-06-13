using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Restores the previous CUDA current device when disposed.
/// 在释放时恢复之前的 CUDA 当前设备。
/// </summary>
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

    /// <summary>
    /// Gets the CUDA device that is active inside this scope.
    /// 获取该作用域内当前激活的 CUDA 设备。
    /// </summary>
    public int CurrentDevice { get; }

    /// <summary>
    /// Gets the CUDA device that was active before the scope was created.
    /// 获取创建该作用域之前处于激活状态的 CUDA 设备。
    /// </summary>
    public int PreviousDevice => _previousDevice;

    /// <summary>
    /// Restores the previous CUDA current device.
    /// 恢复之前的 CUDA 当前设备。
    /// </summary>
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
