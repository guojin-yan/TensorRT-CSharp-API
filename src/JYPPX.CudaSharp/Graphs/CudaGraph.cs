using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a captured CUDA graph.
/// 已捕获 CUDA graph 的托管封装。
/// </summary>
public sealed partial class CudaGraph : IDisposable
{
    private readonly SafeCudaGraphHandle _handle;
    private readonly object _captureLifecycleGate = new object();
    private int _activeCaptureToGraphSessions;
    private int _activeConditionalOwners;
    private int _activeMemoryAllocationOwners;
    private bool _disposed;

    internal CudaGraph(SafeCudaGraphHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    internal SafeCudaGraphHandle Handle => _handle;

    /// <summary>
    /// Creates an empty CUDA graph.
    /// 创建一个空 CUDA graph。
    /// </summary>
    /// <param name="flags">CUDA graph creation flags. CUDA graph 创建标志。</param>
    /// <returns>A managed CUDA graph wrapper. 托管 CUDA graph 封装。</returns>
    public static CudaGraph Create(uint flags = 0)
    {
        NativeBridgeLoader.EnsureInitialized();
        return new CudaGraph(NativeCudaApi.CreateGraph(flags));
    }

    internal static void ValidateDeviceMemory(CudaMemory memory, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    internal static void ValidatePinnedMemory(CudaPinnedMemory memory, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    internal static void ValidateMemcpyCount(int count, int destinationSize, int sourceSize, string parameterName)
    {
        if (count <= 0 || count > destinationSize || count > sourceSize)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    internal static void ValidateMemsetCount(int count, int destinationSize, string parameterName)
    {
        if (count <= 0 || count > destinationSize)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidateKernelNodeAttribute(CudaGraphKernelNodeAttribute attribute, string parameterName)
    {
        switch (attribute)
        {
            case CudaGraphKernelNodeAttribute.Cooperative:
            case CudaGraphKernelNodeAttribute.Priority:
            case CudaGraphKernelNodeAttribute.ClusterDimension:
            case CudaGraphKernelNodeAttribute.ClusterSchedulingPolicyPreference:
                return;
            default:
                throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static int ToListCapacity(ulong count)
    {
        return count > int.MaxValue ? int.MaxValue : (int)count;
    }

    /// <summary>
    /// Releases the captured CUDA graph handle.
    /// 释放已捕获的 CUDA graph 句柄。
    /// </summary>
    public void Dispose()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeCaptureToGraphSessions != 0)
            {
                throw new InvalidOperationException("The CUDA graph cannot be disposed while a stream-to-graph capture session is active.");
            }

            if (_activeConditionalOwners != 0)
            {
                throw new InvalidOperationException("The CUDA graph cannot be disposed while a conditional handle or node wrapper is active.");
            }

            if (_activeMemoryAllocationOwners != 0)
            {
                throw new InvalidOperationException("The CUDA graph cannot be disposed while a graph memory-allocation wrapper is active.");
            }

            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    internal void EnterCaptureToGraphSession()
    {
        lock (_captureLifecycleGate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CudaGraph));
            }

            _activeCaptureToGraphSessions++;
        }
    }

    internal void ExitCaptureToGraphSession()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeCaptureToGraphSessions > 0)
            {
                _activeCaptureToGraphSessions--;
            }
        }
    }

    internal void EnterConditionalOwner()
    {
        lock (_captureLifecycleGate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CudaGraph));
            }

            _activeConditionalOwners++;
        }
    }

    internal void ExitConditionalOwner()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeConditionalOwners > 0)
            {
                _activeConditionalOwners--;
            }
        }
    }

    internal void EnterMemoryAllocationOwner()
    {
        lock (_captureLifecycleGate)
        {
            ThrowIfDisposedCore();
            _activeMemoryAllocationOwners++;
        }
    }

    internal void ExitMemoryAllocationOwner()
    {
        lock (_captureLifecycleGate)
        {
            if (_activeMemoryAllocationOwners > 0)
            {
                _activeMemoryAllocationOwners--;
            }
        }
    }

    private void ThrowIfDisposed()
    {
        lock (_captureLifecycleGate)
        {
            ThrowIfDisposedCore();
        }
    }

    private void ThrowIfDisposedCore()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaGraph));
        }
    }
}
