using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns the bridge metadata for a CUDA conditional node and its CUDA-owned body graphs.
/// 拥有 CUDA conditional node 及其 CUDA-owned body graph 的 bridge metadata。
/// </summary>
public sealed class CudaGraphConditionalNode : IDisposable
{
    private readonly CudaGraph _owner;
    private readonly SafeCudaGraphConditionalNodeHandle _handle;
    private bool _disposed;

    internal CudaGraphConditionalNode(
        CudaGraph owner,
        SafeCudaGraphConditionalNodeHandle handle,
        CudaGraphNode node,
        CudaGraphConditionalNodeType nodeType)
    {
        _owner = owner;
        _handle = handle;
        Node = node;
        Type = nodeType;
        _owner.EnterConditionalOwner();
    }

    internal SafeCudaGraphConditionalNodeHandle Handle => _handle;

    /// <summary>Gets the graph-owned node token for diagnostics and dependency composition.</summary>
    public CudaGraphNode Node { get; }

    /// <summary>Gets the conditional node kind.</summary>
    public CudaGraphConditionalNodeType Type { get; }

    /// <summary>Gets the number of CUDA-owned body graphs.</summary>
    public uint BodyCount
    {
        get
        {
            ThrowIfDisposed();
            return NativeCudaApi.GetConditionalNodeBodyCount(_handle);
        }
    }

    /// <summary>Gets the number of nodes in one conditional body.</summary>
    public ulong GetBodyNodeCount(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.GetConditionalBodyNodeCount(_handle, bodyIndex);
    }

    /// <summary>Gets the number of root nodes in one conditional body.</summary>
    public ulong GetBodyRootNodeCount(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.GetConditionalBodyRootNodeCount(_handle, bodyIndex);
    }

    /// <summary>Gets the number of dependency edges in one conditional body.</summary>
    public ulong GetBodyEdgeCount(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.GetConditionalBodyEdgeCount(_handle, bodyIndex);
    }

    /// <summary>
    /// Adds an empty node to one CUDA-owned conditional body.
    /// 向一个 CUDA-owned conditional body 添加空节点。
    /// </summary>
    public CudaGraphNode AddEmptyNode(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.AddConditionalBodyEmptyNode(_handle, bodyIndex, default);
    }

    /// <summary>Adds an empty node after a body-local dependency node.</summary>
    public CudaGraphNode AddEmptyNodeAfter(uint bodyIndex, CudaGraphNode dependencyNode)
    {
        ThrowIfDisposed();
        return NativeCudaApi.AddConditionalBodyEmptyNode(_handle, bodyIndex, dependencyNode);
    }

    /// <summary>Releases bridge metadata; CUDA destroys the node with its owning graph.</summary>
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaGraphConditionalNode));
        }
    }
}
