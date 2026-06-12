using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static ulong GetDeviceGraphMemoryAttribute(int device, CudaGraphMemoryAttribute attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_graph_memory_attribute(device, (int)attribute, out ulong value));
        return value;
    }

    public static CudaDeviceGraphMemoryInfo GetDeviceGraphMemoryInfo(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_graph_memory_used_current(device, out ulong usedCurrent));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_graph_memory_used_high(device, out ulong usedHigh));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_graph_memory_reserved_current(device, out ulong reservedCurrent));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_graph_memory_reserved_high(device, out ulong reservedHigh));
        return new CudaDeviceGraphMemoryInfo(device, usedCurrent, usedHigh, reservedCurrent, reservedHigh);
    }

    public static void TrimDeviceGraphMemory(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_trim_graph_memory(device));
    }

    public static void ResetDeviceGraphMemoryHighWatermark(int device, CudaGraphMemoryAttribute attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_reset_graph_memory_high_watermark(device, (int)attribute));
    }

    public static void ResetDeviceGraphMemoryHighWatermarks(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_reset_graph_memory_used_high(device));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_reset_graph_memory_reserved_high(device));
    }

    public static bool IsGraphNodeInGraph(SafeCudaGraphHandle graph, CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_is_in_graph_safe(graph, node.Token, out int isInGraph));
        return isInGraph != 0;
    }

    public static uint GetGraphNodeLocalId(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_local_id_safe(node.Token, out uint nodeId));
        return nodeId;
    }

    public static ulong GetGraphNodeToolsId(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_tools_id_safe(node.Token, out ulong toolsNodeId));
        return toolsNodeId;
    }
}
