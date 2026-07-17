using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around an executable CUDA graph.
/// 可执行 CUDA graph 的托管封装。
/// </summary>
public sealed class CudaGraphExec : IDisposable
{
    private readonly SafeCudaGraphExecHandle _handle;

    internal CudaGraphExec(SafeCudaGraphExecHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    internal SafeCudaGraphExecHandle Handle => _handle;

    /// <summary>
    /// Gets graph executable flags when supported by the CUDA runtime.
    /// 在 CUDA runtime 支持时获取 graph executable flags。
    /// </summary>
    public ulong Flags => NativeCudaApi.GetGraphExecFlags(_handle);

    /// <summary>
    /// Gets the CUDA graph executable identifier when supported by the loaded CUDA runtime.
    /// 在当前 CUDA runtime 支持时获取 CUDA graph executable 标识符。
    /// </summary>
    public uint Id => NativeCudaApi.GetGraphExecId(_handle);

    /// <summary>
    /// Sets whether a node is enabled in this executable graph.
    /// 设置 executable graph 中某个节点是否启用。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <param name="enabled">Whether the node should be enabled. 节点是否启用。</param>
    public void SetNodeEnabled(CudaGraphNode node, bool enabled)
    {
        NativeCudaApi.SetGraphExecNodeEnabled(_handle, node, enabled);
    }

    /// <summary>
    /// Gets whether a node is enabled in this executable graph.
    /// 获取 executable graph 中某个节点是否启用。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <returns>True when the node is enabled. 节点启用时返回 true。</returns>
    public bool GetNodeEnabled(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphExecNodeEnabled(_handle, node);
    }

    /// <summary>
    /// Gets a copied executable-state snapshot for a graph node.
    /// 获取 graph node 在当前 executable graph 中的复制型执行状态快照。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <returns>A snapshot containing node-enabled state and executable flags. 包含节点启用状态和 executable flags 的快照。</returns>
    /// <remarks>
    /// The returned snapshot contains copied scalar values only and does not transfer ownership of the node token.
    /// 返回的快照仅包含复制出的标量值，不转移 node token 的所有权。
    /// </remarks>
    public CudaGraphExecNodeStateSnapshot GetNodeStateSnapshot(CudaGraphNode node)
    {
        return new CudaGraphExecNodeStateSnapshot(GetNodeEnabled(node), Flags);
    }

    /// <summary>
    /// Gets copied executable-state snapshots for multiple graph nodes.
    /// 获取多个 graph node 的复制型 executable 状态快照。
    /// </summary>
    /// <param name="nodes">The graph-owned node value tokens. graph 拥有的 node 值 token。</param>
    /// <returns>A copied executable diagnostic snapshot. 复制型 executable 诊断快照。</returns>
    public CudaGraphExecDiagnosticSnapshot GetDiagnosticSnapshot(IReadOnlyList<CudaGraphNode> nodes)
    {
        if (nodes == null)
        {
            throw new ArgumentNullException(nameof(nodes));
        }

        List<CudaGraphExecNodeStateSnapshot> snapshots = new List<CudaGraphExecNodeStateSnapshot>(nodes.Count);
        for (int index = 0; index < nodes.Count; index++)
        {
            CudaGraphNode node = nodes[index];
            snapshots.Add(new CudaGraphExecNodeStateSnapshot((ulong)index, node, GetNodeEnabled(node), Flags));
        }

        return new CudaGraphExecDiagnosticSnapshot(snapshots);
    }

    /// <summary>
    /// Updates an executable graph event-record node to use a caller-owned CUDA event.
    /// 将 executable graph 的 event-record 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while this executable graph may use the updated node.
    /// 该 event 必须在当前 executable graph 可能使用更新节点期间保持存活。
    /// </remarks>
    public void SetEventRecordNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphExecEventRecordNodeEvent(_handle, node, eventHandle.Handle);
    }

    /// <summary>
    /// Updates an executable graph event-wait node to use a caller-owned CUDA event.
    /// 将 executable graph 的 event-wait 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while this executable graph may use the updated node.
    /// 该 event 必须在当前 executable graph 可能使用更新节点期间保持存活。
    /// </remarks>
    public void SetEventWaitNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphExecEventWaitNodeEvent(_handle, node, eventHandle.Handle);
    }

    /// <summary>
    /// Updates an executable graph memcpy node for a 1D device-to-device copy.
    /// 将 executable graph 的 memcpy 节点更新为 1D device-to-device 复制。
    /// </summary>
    /// <param name="node">A memcpy node token owned by the source graph. 源 graph 拥有的 memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// The memory owners must remain alive while this executable graph may use the updated node.
    /// 内存所有者必须在当前 executable graph 可能使用更新节点期间保持存活。
    /// </remarks>
    public void SetDeviceToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaMemory source, int count)
    {
        CudaGraph.ValidateDeviceMemory(destination, nameof(destination));
        CudaGraph.ValidateDeviceMemory(source, nameof(source));
        CudaGraph.ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphExecMemcpyNodeParametersDeviceToDevice(_handle, node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates an executable graph memcpy node for a 1D pinned-host-to-device copy.
    /// 将 executable graph 的 memcpy 节点更新为 1D pinned-host-to-device 复制。
    /// </summary>
    /// <param name="node">A memcpy node token owned by the source graph. 源 graph 拥有的 memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// The memory owners must remain alive while this executable graph may use the updated node.
    /// 内存所有者必须在当前 executable graph 可能使用更新节点期间保持存活。
    /// </remarks>
    public void SetHostToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaPinnedMemory source, int count)
    {
        CudaGraph.ValidateDeviceMemory(destination, nameof(destination));
        CudaGraph.ValidatePinnedMemory(source, nameof(source));
        CudaGraph.ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphExecMemcpyNodeParametersHostToDevice(_handle, node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates an executable graph memcpy node for a 1D device-to-pinned-host copy.
    /// 将 executable graph 的 memcpy 节点更新为 1D device-to-pinned-host 复制。
    /// </summary>
    /// <param name="node">A memcpy node token owned by the source graph. 源 graph 拥有的 memcpy 节点 token。</param>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// The memory owners must remain alive while this executable graph may use the updated node.
    /// 内存所有者必须在当前 executable graph 可能使用更新节点期间保持存活。
    /// </remarks>
    public void SetDeviceToHostMemcpyNodeParameters(CudaGraphNode node, CudaPinnedMemory destination, CudaMemory source, int count)
    {
        CudaGraph.ValidatePinnedMemory(destination, nameof(destination));
        CudaGraph.ValidateDeviceMemory(source, nameof(source));
        CudaGraph.ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphExecMemcpyNodeParametersDeviceToHost(_handle, node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Replaces the parameters of a child-graph node using a managed graph owner.
    /// 使用托管 graph owner 替换 child-graph 节点参数。
    /// </summary>
    /// <param name="node">A child-graph node token from the source graph. 源 graph 中的 child-graph 节点 token。</param>
    /// <param name="childGraph">The graph supplying replacement parameters. 提供替换参数的 graph。</param>
    public void SetChildGraphNodeParameters(CudaGraphNode node, CudaGraph childGraph)
    {
        if (childGraph == null)
        {
            throw new ArgumentNullException(nameof(childGraph));
        }

        NativeCudaApi.SetGraphExecChildGraphNodeParameters(_handle, node, childGraph.Handle);
    }

    /// <summary>
    /// Replaces a one-dimensional memset node using a managed device-memory owner.
    /// 使用托管设备内存 owner 替换一维 memset 节点参数。
    /// </summary>
    /// <remarks>The destination must remain alive while this executable graph may use the node. 当 executable graph 可能使用该节点时，destination 必须保持存活。</remarks>
    public void SetMemsetNodeParameters(CudaGraphNode node, CudaMemory destination, byte value, int count)
    {
        CudaGraph.ValidateDeviceMemory(destination, nameof(destination));
        CudaGraph.ValidateMemsetCount(count, destination.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphExecMemsetNodeParameters(_handle, node, destination.Handle, value, count);
    }

    /// <summary>
    /// Attempts to update this executable graph and returns copied result metadata.
    /// 尝试更新当前 executable graph，并返回复制出的结果元数据。
    /// </summary>
    /// <param name="graph">The graph containing updated parameters. 包含更新参数的 graph。</param>
    /// <returns>A pointer-free update snapshot. 不含指针的更新快照。</returns>
    public CudaGraphExecUpdateSnapshot Update(CudaGraph graph)
    {
        if (graph == null)
        {
            throw new ArgumentNullException(nameof(graph));
        }

        return NativeCudaApi.UpdateGraphExec(_handle, graph.Handle);
    }

    /// <summary>
    /// Uploads this executable graph to a CUDA stream before launch.
    /// 在 launch 前将当前可执行 graph 上传到 CUDA stream。
    /// </summary>
    /// <param name="stream">The stream used for graph upload. 用于 graph upload 的 stream。</param>
    public void Upload(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.UploadGraphExec(_handle, stream.Handle);
    }

    /// <summary>
    /// Launches the executable graph on a CUDA stream.
    /// 在 CUDA stream 上启动当前可执行 graph。
    /// </summary>
    /// <param name="stream">The stream used for graph launch. 用于 graph 启动的 stream。</param>
    public void Launch(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.LaunchGraphExec(_handle, stream.Handle);
    }

    /// <summary>
    /// Releases the executable graph handle.
    /// 释放可执行 graph 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
