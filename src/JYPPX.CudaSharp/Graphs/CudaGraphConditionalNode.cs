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

    /// <summary>Gets the graph-owned node token for diagnostics and dependency composition. 获取用于诊断和依赖组合的 graph-owned 节点 token。</summary>
    public CudaGraphNode Node { get; }

    /// <summary>Gets the conditional node kind. 获取条件节点类型。</summary>
    public CudaGraphConditionalNodeType Type { get; }

    /// <summary>Gets the number of CUDA-owned body graphs. 获取 CUDA-owned body graph 的数量。</summary>
    public uint BodyCount
    {
        get
        {
            ThrowIfDisposed();
            return NativeCudaApi.GetConditionalNodeBodyCount(_handle);
        }
    }

    /// <summary>Gets the number of nodes in one conditional body. 获取一个 conditional body 中的节点数量。</summary>
    public ulong GetBodyNodeCount(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.GetConditionalBodyNodeCount(_handle, bodyIndex);
    }

    /// <summary>Gets the number of root nodes in one conditional body. 获取一个 conditional body 中的根节点数量。</summary>
    public ulong GetBodyRootNodeCount(uint bodyIndex)
    {
        ThrowIfDisposed();
        return NativeCudaApi.GetConditionalBodyRootNodeCount(_handle, bodyIndex);
    }

    /// <summary>Gets the number of dependency edges in one conditional body. 获取一个 conditional body 中的依赖边数量。</summary>
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

    /// <summary>Adds an empty node after a body-local dependency node. 在 body-local 依赖节点后添加空节点。</summary>
    public CudaGraphNode AddEmptyNodeAfter(uint bodyIndex, CudaGraphNode dependencyNode)
    {
        ThrowIfDisposed();
        return NativeCudaApi.AddConditionalBodyEmptyNode(_handle, bodyIndex, dependencyNode);
    }

    /// <summary>Releases bridge metadata; CUDA destroys the node with its owning graph. 释放 bridge 元数据；CUDA 随所属 graph 销毁节点。</summary>
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
