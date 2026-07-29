using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Clones this CUDA graph into a new managed graph wrapper.
    /// 将当前 CUDA graph 克隆为新的托管 graph 封装。
    /// </summary>
    /// <returns>A cloned CUDA graph. 克隆后的 CUDA graph。</returns>
    public CudaGraph Clone()
    {
        return new CudaGraph(NativeCudaApi.CloneGraph(_handle));
    }

    /// <summary>
    /// Adds a child-graph node containing a clone of another managed CUDA graph.
    /// 添加一个包含另一托管 CUDA graph 克隆的 child-graph 节点。
    /// </summary>
    /// <param name="childGraph">The graph copied into the new node. 复制到新节点中的 graph。</param>
    /// <returns>A graph-owned child node token. 由当前 graph 拥有的 child node token。</returns>
    public CudaGraphNode AddChildGraphNode(CudaGraph childGraph)
    {
        if (childGraph == null)
        {
            throw new ArgumentNullException(nameof(childGraph));
        }

        return NativeCudaApi.AddGraphChildGraphNode(_handle, childGraph.Handle);
    }

    /// <summary>
    /// Adds a child-graph node after an existing dependency node.
    /// 在现有依赖节点之后添加 child-graph 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <param name="childGraph">The graph copied into the new node. 复制到新节点中的 graph。</param>
    /// <returns>A graph-owned child node token. 由当前 graph 拥有的 child node token。</returns>
    public CudaGraphNode AddChildGraphNodeAfter(CudaGraphNode dependencyNode, CudaGraph childGraph)
    {
        if (childGraph == null)
        {
            throw new ArgumentNullException(nameof(childGraph));
        }

        return NativeCudaApi.AddGraphChildGraphNodeAfter(_handle, dependencyNode, childGraph.Handle);
    }

    /// <summary>
    /// Adds a CUDA conditional node with no parent dependency.
    /// 添加一个没有 parent dependency 的 CUDA conditional node。
    /// </summary>
    public CudaGraphConditionalNode AddConditionalNode(
        CudaGraphConditionalHandle handle,
        CudaGraphConditionalNodeType nodeType,
        uint bodyCount)
    {
        return AddConditionalNodeAfter(handle, nodeType, bodyCount, default);
    }

    /// <summary>
    /// Adds a CUDA conditional node after a parent graph dependency.
    /// 在 parent graph dependency 之后添加 CUDA conditional node。
    /// </summary>
    public CudaGraphConditionalNode AddConditionalNodeAfter(
        CudaGraphConditionalHandle handle,
        CudaGraphConditionalNodeType nodeType,
        uint bodyCount,
        CudaGraphNode dependencyNode)
    {
        ThrowIfDisposed();
        if (handle == null)
        {
            throw new ArgumentNullException(nameof(handle));
        }

        if (!ReferenceEquals(handle.Owner, this))
        {
            throw new ArgumentException("The conditional handle belongs to a different CUDA graph.", nameof(handle));
        }

        if (handle.IsDisposed)
        {
            throw new ObjectDisposedException(nameof(handle));
        }

        if (bodyCount == 0 || (nodeType == CudaGraphConditionalNodeType.If && bodyCount > 2) ||
            (nodeType == CudaGraphConditionalNodeType.While && bodyCount != 1))
        {
            throw new ArgumentOutOfRangeException(nameof(bodyCount));
        }

        SafeCudaGraphConditionalNodeHandle nativeNode = NativeCudaApi.AddConditionalNode(
            _handle,
            handle.Handle,
            nodeType,
            bodyCount,
            dependencyNode);
        try
        {
            return new CudaGraphConditionalNode(
                this,
                nativeNode,
                NativeCudaApi.GetConditionalNodeToken(nativeNode),
                nodeType);
        }
        catch
        {
            nativeNode.Dispose();
            throw;
        }
    }

}
