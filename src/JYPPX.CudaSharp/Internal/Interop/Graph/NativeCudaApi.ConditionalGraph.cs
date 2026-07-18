using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaGraphConditionalHandleHandle CreateConditionalHandle(
        SafeCudaGraphHandle graph,
        uint defaultLaunchValue,
        CudaGraphConditionalHandleFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_handle_create_safe(
            graph,
            defaultLaunchValue,
            (uint)flags,
            out SafeCudaGraphConditionalHandleHandle handle));
        return handle;
    }

    public static SafeCudaGraphConditionalHandleHandle CreateConditionalHandleV2(
        SafeCudaGraphHandle graph,
        SafeCudaExecutionContextHandle? context,
        uint defaultLaunchValue,
        CudaGraphConditionalHandleFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_handle_create_v2_safe(
            graph,
            context,
            defaultLaunchValue,
            (uint)flags,
            out SafeCudaGraphConditionalHandleHandle handle));
        return handle;
    }

    public static SafeCudaGraphConditionalNodeHandle AddConditionalNode(
        SafeCudaGraphHandle graph,
        SafeCudaGraphConditionalHandleHandle handle,
        CudaGraphConditionalNodeType nodeType,
        uint bodyCount,
        CudaGraphNode dependencyNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_conditional_node_safe(
            graph,
            handle,
            (uint)nodeType,
            bodyCount,
            dependencyNode.Token,
            out SafeCudaGraphConditionalNodeHandle node));
        return node;
    }

    public static uint GetConditionalNodeBodyCount(SafeCudaGraphConditionalNodeHandle node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_node_get_body_count_safe(
            node,
            out uint bodyCount));
        return bodyCount;
    }

    public static CudaGraphNode GetConditionalNodeToken(SafeCudaGraphConditionalNodeHandle node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_node_get_node_token_safe(
            node,
            out UIntPtr nodeToken));
        return new CudaGraphNode(nodeToken);
    }

    public static ulong GetConditionalBodyNodeCount(SafeCudaGraphConditionalNodeHandle node, uint bodyIndex)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_body_get_node_count_safe(
            node,
            bodyIndex,
            out UIntPtr count));
        return count.ToUInt64();
    }

    public static ulong GetConditionalBodyRootNodeCount(SafeCudaGraphConditionalNodeHandle node, uint bodyIndex)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_body_get_root_node_count_safe(
            node,
            bodyIndex,
            out UIntPtr count));
        return count.ToUInt64();
    }

    public static ulong GetConditionalBodyEdgeCount(SafeCudaGraphConditionalNodeHandle node, uint bodyIndex)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_body_get_edge_count_safe(
            node,
            bodyIndex,
            out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNode AddConditionalBodyEmptyNode(
        SafeCudaGraphConditionalNodeHandle node,
        uint bodyIndex,
        CudaGraphNode dependencyNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_conditional_body_add_empty_node_safe(
            node,
            bodyIndex,
            dependencyNode.Token,
            out UIntPtr bodyNode));
        return new CudaGraphNode(bodyNode);
    }
}
