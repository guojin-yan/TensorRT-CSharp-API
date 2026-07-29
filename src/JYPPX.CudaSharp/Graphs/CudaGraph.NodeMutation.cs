using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Updates a CUDA graph memset node using a managed device-memory owner and byte-count descriptor.
    /// 使用托管设备内存所有者和字节数 descriptor 更新 CUDA graph memset 节点。
    /// </summary>
    /// <param name="node">The memset node token. memset 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="value">The byte value used by the memset node. memset 节点使用的字节填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    /// <remarks>
    /// This method does not accept raw device pointers. The native bridge derives the destination pointer from <paramref name="destination"/> and validates <paramref name="count"/> against the allocation size.
    /// 该方法不接受裸设备指针。Native bridge 会从 <paramref name="destination"/> 推导目标指针，并按分配大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetMemsetNodeParameters(CudaGraphNode node, CudaMemory destination, byte value, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (count <= 0 || count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.SetGraphMemsetNodeParameters(node, destination.Handle, value, count);
    }

    /// <summary>
    /// Updates a CUDA graph kernel node scalar attribute using a typed descriptor.
    /// 使用 typed descriptor 更新 CUDA graph kernel 节点的标量 attribute。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="value">The scalar attribute descriptor. 标量 attribute descriptor。</param>
    /// <remarks>
    /// This method does not accept raw kernel parameters, function pointers, callback user data, or CUDA's native union.
    /// 该方法不接受裸 kernel 参数、function pointer、callback user data 或 CUDA 原生 union。
    /// </remarks>
    public static void SetKernelNodeAttribute(CudaGraphNode node, CudaGraphKernelNodeAttributeValue value)
    {
        ValidateKernelNodeAttribute(value.Attribute, nameof(value));
        NativeCudaApi.SetGraphKernelNodeAttribute(node, value);
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cooperative flag.
    /// 更新 CUDA graph kernel 节点 cooperative 标志。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="enabled">Whether cooperative launch is enabled. 是否启用 cooperative launch。</param>
    public static void SetKernelNodeCooperative(CudaGraphNode node, bool enabled)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.Cooperative(enabled));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node priority.
    /// 更新 CUDA graph kernel 节点 priority。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="priority">The kernel execution priority. Kernel 执行优先级。</param>
    public static void SetKernelNodePriority(CudaGraphNode node, int priority)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.Priority(priority));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cluster dimension.
    /// 更新 CUDA graph kernel 节点 cluster dimension。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="x">The cluster X dimension. Cluster X 维度。</param>
    /// <param name="y">The cluster Y dimension. Cluster Y 维度。</param>
    /// <param name="z">The cluster Z dimension. Cluster Z 维度。</param>
    public static void SetKernelNodeClusterDimension(CudaGraphNode node, uint x, uint y, uint z)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.ClusterDimension(x, y, z));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cluster scheduling policy preference.
    /// 更新 CUDA graph kernel 节点 cluster scheduling policy preference。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="policy">The cluster scheduling policy preference. Cluster 调度策略偏好。</param>
    public static void SetKernelNodeClusterSchedulingPolicy(CudaGraphNode node, CudaClusterSchedulingPolicyPreference policy)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.ClusterSchedulingPolicy(policy));
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D device-to-device copy.
    /// 将 CUDA graph memcpy 节点更新为 1D device-to-device 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetDeviceToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersDeviceToDevice(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D pinned-host-to-device copy.
    /// 将 CUDA graph memcpy 节点更新为 1D pinned-host-to-device 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetHostToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaPinnedMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidatePinnedMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersHostToDevice(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D device-to-pinned-host copy.
    /// 将 CUDA graph memcpy 节点更新为 1D device-to-pinned-host 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetDeviceToHostMemcpyNodeParameters(CudaGraphNode node, CudaPinnedMemory destination, CudaMemory source, int count)
    {
        ValidatePinnedMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersDeviceToHost(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates an event-record node to use a caller-owned CUDA event.
    /// 将 event-record 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">The event-record node token. event-record 节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public static void SetEventRecordNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphEventRecordNodeEvent(node, eventHandle.Handle);
    }

    /// <summary>
    /// Gets whether an event-record node currently references a CUDA event without exposing the borrowed event handle.
    /// 查询 event-record 节点当前是否引用 CUDA event，但不暴露 borrowed event handle。
    /// </summary>
    /// <param name="node">The graph-owned event-record node token. Graph 拥有的 event-record node token。</param>
    /// <returns><see langword="true"/> when CUDA reports a non-null event for the node. CUDA 报告该节点具有非空 event 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is a copied scalar snapshot over <c>cudaGraphEventRecordNodeGetEvent</c>; it intentionally does not return <see cref="CudaEvent"/> or a native pointer.
    /// 这是基于 <c>cudaGraphEventRecordNodeGetEvent</c> 的复制型标量快照；有意不返回 <see cref="CudaEvent"/> 或原生指针。
    /// </remarks>
    public static bool EventRecordNodeHasEvent(CudaGraphNode node)
    {
        return NativeCudaApi.GraphEventRecordNodeHasEvent(node);
    }

    /// <summary>
    /// Updates an event-wait node to use a caller-owned CUDA event.
    /// 将 event-wait 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">The event-wait node token. event-wait 节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public static void SetEventWaitNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphEventWaitNodeEvent(node, eventHandle.Handle);
    }

    /// <summary>
    /// Gets whether an event-wait node currently references a CUDA event without exposing the borrowed event handle.
    /// 查询 event-wait 节点当前是否引用 CUDA event，但不暴露 borrowed event handle。
    /// </summary>
    /// <param name="node">The graph-owned event-wait node token. Graph 拥有的 event-wait node token。</param>
    /// <returns><see langword="true"/> when CUDA reports a non-null event for the node. CUDA 报告该节点具有非空 event 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is a copied scalar snapshot over <c>cudaGraphEventWaitNodeGetEvent</c>; it intentionally does not return <see cref="CudaEvent"/> or a native pointer.
    /// 这是基于 <c>cudaGraphEventWaitNodeGetEvent</c> 的复制型标量快照；有意不返回 <see cref="CudaEvent"/> 或原生指针。
    /// </remarks>
    public static bool EventWaitNodeHasEvent(CudaGraphNode node)
    {
        return NativeCudaApi.GraphEventWaitNodeHasEvent(node);
    }

}
