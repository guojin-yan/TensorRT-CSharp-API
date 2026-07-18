using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns a CUDA conditional graph handle associated with one <see cref="CudaGraph"/>.
/// 拥有一个与指定 <see cref="CudaGraph"/> 关联的 CUDA conditional graph handle。
/// </summary>
public sealed class CudaGraphConditionalHandle : IDisposable
{
    private readonly CudaGraph _owner;
    private readonly SafeCudaGraphConditionalHandleHandle _handle;
    private bool _disposed;

    internal CudaGraphConditionalHandle(
        CudaGraph owner,
        SafeCudaGraphConditionalHandleHandle handle,
        uint defaultLaunchValue,
        CudaGraphConditionalHandleFlags flags)
    {
        _owner = owner;
        _handle = handle;
        DefaultLaunchValue = defaultLaunchValue;
        Flags = flags;
        _owner.EnterConditionalOwner();
    }

    internal SafeCudaGraphConditionalHandleHandle Handle => _handle;

    internal CudaGraph Owner => _owner;

    internal bool IsDisposed => _disposed;

    /// <summary>Gets the default conditional value supplied at creation.</summary>
    public uint DefaultLaunchValue { get; }

    /// <summary>Gets the creation flags supplied to CUDA.</summary>
    public CudaGraphConditionalHandleFlags Flags { get; }

    /// <summary>
    /// Releases the bridge metadata. CUDA owns the underlying conditional value until its graph is destroyed.
    /// 释放 bridge metadata；底层 conditional value 在 graph 销毁前由 CUDA 持有。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            _handle.Dispose();
        }
        finally
        {
            _owner.ExitConditionalOwner();
        }

        GC.SuppressFinalize(this);
    }
}
