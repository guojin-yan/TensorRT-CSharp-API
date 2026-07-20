using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaGraphMemoryAllocationHandle AddGraphMemoryAllocationNode(
        SafeCudaGraphHandle graph,
        CudaGraphNode dependencyNode,
        int sizeInBytes,
        int deviceOrdinal)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memory_allocation_create_safe(
            graph,
            dependencyNode.Token,
            new UIntPtr((uint)sizeInBytes),
            deviceOrdinal,
            out SafeCudaGraphMemoryAllocationHandle allocation));
        return allocation;
    }

    public static CudaGraphNode AddGraphMemoryAllocationMemsetNode(
        SafeCudaGraphHandle graph,
        SafeCudaGraphMemoryAllocationHandle allocation,
        CudaGraphNode dependencyNode,
        byte value,
        int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memory_allocation_add_memset_node_safe(
            graph,
            allocation,
            dependencyNode.Token,
            value,
            new UIntPtr((uint)count),
            out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemoryAllocationDeviceToHostNode(
        SafeCudaGraphHandle graph,
        SafeCudaGraphMemoryAllocationHandle allocation,
        CudaGraphNode dependencyNode,
        SafeCudaPinnedMemoryHandle destination,
        int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memory_allocation_add_device_to_host_node_safe(
            graph,
            allocation,
            dependencyNode.Token,
            destination,
            new UIntPtr((uint)count),
            out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemoryFreeNode(
        SafeCudaGraphHandle graph,
        SafeCudaGraphMemoryAllocationHandle allocation,
        CudaGraphNode dependencyNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memory_allocation_add_free_node_safe(
            graph,
            allocation,
            dependencyNode.Token,
            out UIntPtr node));
        return new CudaGraphNode(node);
    }
}
