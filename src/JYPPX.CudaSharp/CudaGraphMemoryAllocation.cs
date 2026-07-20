using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns pointer-free bridge metadata for a CUDA graph memory-allocation node.
/// 拥有 CUDA graph memory-allocation node 的无指针 bridge 元数据。
/// </summary>
/// <remarks>
/// The allocation exists only while an executable graph is running. This wrapper never exposes the
/// CUDA device address and can only be consumed by its owning <see cref="CudaGraph"/>.
/// 该分配只在 executable graph 执行期间存在。本 wrapper 不公开 CUDA device address，且只能由所属
/// <see cref="CudaGraph"/> 使用。
/// </remarks>
public sealed class CudaGraphMemoryAllocation : IDisposable
{
    private readonly object _stateGate = new object();
    private readonly CudaGraph _owner;
    private readonly SafeCudaGraphMemoryAllocationHandle _handle;
    private bool _freeNodeAdded;
    private bool _disposed;

    internal CudaGraphMemoryAllocation(
        CudaGraph owner,
        SafeCudaGraphMemoryAllocationHandle handle,
        int sizeInBytes,
        int deviceOrdinal)
    {
        _owner = owner ?? throw new ArgumentNullException(nameof(owner));
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        SizeInBytes = sizeInBytes;
        DeviceOrdinal = deviceOrdinal;
        _owner.EnterMemoryAllocationOwner();
    }

    /// <summary>Gets the requested allocation size in bytes. 获取请求的分配字节数。</summary>
    public int SizeInBytes { get; }

    /// <summary>Gets the CUDA device ordinal used by the allocation node. 获取 allocation node 使用的 CUDA device ordinal。</summary>
    public int DeviceOrdinal { get; }

    /// <summary>Gets whether a matching memory-free node has been added. 获取是否已添加匹配的 memory-free node。</summary>
    public bool IsFreeNodeAdded
    {
        get
        {
            lock (_stateGate)
            {
                return _freeNodeAdded;
            }
        }
    }

    /// <summary>Gets whether this bridge metadata wrapper has been disposed. 获取 bridge 元数据 wrapper 是否已释放。</summary>
    public bool IsDisposed
    {
        get
        {
            lock (_stateGate)
            {
                return _disposed;
            }
        }
    }

    internal T UseBeforeFree<T>(CudaGraph owner, Func<SafeCudaGraphMemoryAllocationHandle, T> action)
    {
        if (!ReferenceEquals(owner, _owner))
        {
            throw new ArgumentException("The graph memory allocation belongs to a different CUDA graph.", nameof(owner));
        }
        if (action == null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        lock (_stateGate)
        {
            ThrowIfUnavailable();
            return action(_handle);
        }
    }

    internal CudaGraphNode AddFreeNode(
        CudaGraph owner,
        Func<SafeCudaGraphMemoryAllocationHandle, CudaGraphNode> action)
    {
        if (!ReferenceEquals(owner, _owner))
        {
            throw new ArgumentException("The graph memory allocation belongs to a different CUDA graph.", nameof(owner));
        }
        if (action == null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        lock (_stateGate)
        {
            ThrowIfUnavailable();
            CudaGraphNode node = action(_handle);
            _freeNodeAdded = true;
            return node;
        }
    }

    /// <summary>
    /// Releases bridge metadata. CUDA graph nodes remain owned by the graph.
    /// 释放 bridge 元数据；CUDA graph nodes 仍由 graph 拥有。
    /// </summary>
    public void Dispose()
    {
        lock (_stateGate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        try
        {
            _handle.Dispose();
        }
        finally
        {
            _owner.ExitMemoryAllocationOwner();
        }

        GC.SuppressFinalize(this);
    }

    private void ThrowIfUnavailable()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaGraphMemoryAllocation));
        }
        if (_freeNodeAdded)
        {
            throw new InvalidOperationException("A memory-free node has already been added for this CUDA graph allocation.");
        }
    }
}
