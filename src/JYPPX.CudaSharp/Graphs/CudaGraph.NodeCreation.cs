using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Adds an empty node to this graph.
    /// 向当前 graph 添加一个空节点。
    /// </summary>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode AddEmptyNode()
    {
        return NativeCudaApi.AddGraphEmptyNode(_handle);
    }

    /// <summary>
    /// Adds an empty node that depends on an existing node.
    /// 添加一个依赖现有节点的空节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode AddEmptyNodeAfter(CudaGraphNode dependencyNode)
    {
        return NativeCudaApi.AddGraphEmptyNodeAfter(_handle, dependencyNode);
    }

    /// <summary>
    /// Adds a pointer-free, graph-owned device-memory allocation node.
    /// 添加无指针、由 graph 拥有的 device-memory allocation node。
    /// </summary>
    public CudaGraphMemoryAllocation AddMemoryAllocationNode(int sizeInBytes, int deviceOrdinal)
    {
        return AddMemoryAllocationNodeAfter(default, sizeInBytes, deviceOrdinal);
    }

    /// <summary>
    /// Adds a dependent pointer-free graph memory-allocation node.
    /// 添加带依赖的无指针 graph memory-allocation node。
    /// </summary>
    public CudaGraphMemoryAllocation AddMemoryAllocationNodeAfter(
        CudaGraphNode dependencyNode,
        int sizeInBytes,
        int deviceOrdinal)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }
        if (deviceOrdinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceOrdinal));
        }

        lock (_captureLifecycleGate)
        {
            ThrowIfDisposedCore();
            SafeCudaGraphMemoryAllocationHandle handle = NativeCudaApi.AddGraphMemoryAllocationNode(
                _handle,
                dependencyNode,
                sizeInBytes,
                deviceOrdinal);
            try
            {
                return new CudaGraphMemoryAllocation(this, handle, sizeInBytes, deviceOrdinal);
            }
            catch
            {
                handle.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// Adds a one-dimensional byte memset node backed by a managed device-memory owner.
    /// 添加由托管设备内存 owner 支撑的一维 byte memset 节点。
    /// </summary>
    /// <remarks>The destination must remain alive while this graph or an executable graph may use the node. 当 graph 或 executable graph 可能使用该节点时，destination 必须保持存活。</remarks>
    public CudaGraphNode AddMemsetNode(CudaMemory destination, byte value, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateMemsetCount(count, destination.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemsetNode(_handle, destination.Handle, value, count);
    }

    /// <summary>
    /// Adds a one-dimensional byte memset node after an existing dependency.
    /// 在现有 dependency 之后添加一维 byte memset 节点。
    /// </summary>
    /// <remarks>The destination must remain alive while this graph or an executable graph may use the node. 当 graph 或 executable graph 可能使用该节点时，destination 必须保持存活。</remarks>
    public CudaGraphNode AddMemsetNodeAfter(CudaGraphNode dependencyNode, CudaMemory destination, byte value, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateMemsetCount(count, destination.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemsetNodeAfter(_handle, dependencyNode, destination.Handle, value, count);
    }

    /// <summary>
    /// Adds a memset node for a pointer-free graph allocation and depends on its allocation node.
    /// 为无指针 graph allocation 添加 memset node，并自动依赖其 allocation node。
    /// </summary>
    public CudaGraphNode AddMemsetNode(CudaGraphMemoryAllocation destination, byte value, int count)
    {
        return AddMemsetNodeAfter(default, destination, value, count);
    }

    /// <summary>Adds a dependent memset node for a graph allocation. 为 graph allocation 添加带依赖的 memset node。</summary>
    public CudaGraphNode AddMemsetNodeAfter(
        CudaGraphNode dependencyNode,
        CudaGraphMemoryAllocation destination,
        byte value,
        int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }
        if (count <= 0 || count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        return destination.UseBeforeFree(
            this,
            allocation => NativeCudaApi.AddGraphMemoryAllocationMemsetNode(
                _handle,
                allocation,
                dependencyNode,
                value,
                count));
    }

    /// <summary>
    /// Adds an event-record node to this graph.
    /// 向当前 graph 添加 event-record 节点。
    /// </summary>
    /// <param name="eventHandle">The caller-owned CUDA event used by the graph node. graph 节点使用的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-record node token. 由 graph 拥有的 event-record node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventRecordNode(CudaEvent eventHandle)
    {
        return AddEventRecordNodeAfter(default, eventHandle);
    }

    /// <summary>
    /// Adds an event-record node that depends on an existing graph node.
    /// 添加一个依赖现有节点的 event-record 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <param name="eventHandle">The caller-owned CUDA event used by the graph node. graph 节点使用的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-record node token. 由 graph 拥有的 event-record node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventRecordNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        return NativeCudaApi.AddGraphEventRecordNode(_handle, dependencyNode, eventHandle.Handle);
    }

    /// <summary>
    /// Adds an event-wait node to this graph.
    /// 向当前 graph 添加 event-wait 节点。
    /// </summary>
    /// <param name="eventHandle">The caller-owned CUDA event waited on by the graph node. graph 节点等待的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-wait node token. 由 graph 拥有的 event-wait node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventWaitNode(CudaEvent eventHandle)
    {
        return AddEventWaitNodeAfter(default, eventHandle);
    }

    /// <summary>
    /// Adds an event-wait node that depends on an existing graph node.
    /// 添加一个依赖现有节点的 event-wait 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <param name="eventHandle">The caller-owned CUDA event waited on by the graph node. graph 节点等待的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-wait node token. 由 graph 拥有的 event-wait node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventWaitNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        return NativeCudaApi.AddGraphEventWaitNode(_handle, dependencyNode, eventHandle.Handle);
    }

    /// <summary>
    /// Adds a 1D device-to-device memcpy node to this graph.
    /// 向当前 graph 添加 1D device-to-device memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddDeviceToDeviceMemcpyNode(CudaMemory destination, CudaMemory source, int count)
    {
        return AddDeviceToDeviceMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D device-to-device memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D device-to-device memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddDeviceToDeviceMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaMemory destination, CudaMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DDeviceToDevice(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Adds a 1D pinned-host-to-device memcpy node to this graph.
    /// 向当前 graph 添加 1D pinned-host-to-device memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddHostToDeviceMemcpyNode(CudaMemory destination, CudaPinnedMemory source, int count)
    {
        return AddHostToDeviceMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D pinned-host-to-device memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D pinned-host-to-device memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddHostToDeviceMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaMemory destination, CudaPinnedMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidatePinnedMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DHostToDevice(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Adds a 1D device-to-pinned-host memcpy node to this graph.
    /// 向当前 graph 添加 1D device-to-pinned-host memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddDeviceToHostMemcpyNode(CudaPinnedMemory destination, CudaMemory source, int count)
    {
        return AddDeviceToHostMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D device-to-pinned-host memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D device-to-pinned-host memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddDeviceToHostMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaPinnedMemory destination, CudaMemory source, int count)
    {
        ValidatePinnedMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DDeviceToHost(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Adds a graph-allocation-to-pinned-host copy node for runtime verification or output transfer.
    /// 添加 graph allocation 到 pinned host 的复制节点，用于运行验证或输出传输。
    /// </summary>
    public CudaGraphNode AddDeviceToHostMemcpyNode(
        CudaPinnedMemory destination,
        CudaGraphMemoryAllocation source,
        int count)
    {
        return AddDeviceToHostMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>Adds a dependent graph-allocation-to-host copy node. 添加带依赖的 graph allocation 到 host 复制节点。</summary>
    public CudaGraphNode AddDeviceToHostMemcpyNodeAfter(
        CudaGraphNode dependencyNode,
        CudaPinnedMemory destination,
        CudaGraphMemoryAllocation source,
        int count)
    {
        ValidatePinnedMemory(destination, nameof(destination));
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }
        if (count <= 0 || count > source.SizeInBytes || count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        return source.UseBeforeFree(
            this,
            allocation => NativeCudaApi.AddGraphMemoryAllocationDeviceToHostNode(
                _handle,
                allocation,
                dependencyNode,
                destination.Handle,
                count));
    }

    /// <summary>
    /// Adds the single matching memory-free node for a graph allocation.
    /// 为 graph allocation 添加唯一匹配的 memory-free node。
    /// </summary>
    public CudaGraphNode AddMemoryFreeNode(
        CudaGraphMemoryAllocation allocation,
        CudaGraphNode dependencyNode = default)
    {
        if (allocation == null)
        {
            throw new ArgumentNullException(nameof(allocation));
        }

        return allocation.AddFreeNode(
            this,
            handle => NativeCudaApi.AddGraphMemoryFreeNode(_handle, handle, dependencyNode));
    }

}
