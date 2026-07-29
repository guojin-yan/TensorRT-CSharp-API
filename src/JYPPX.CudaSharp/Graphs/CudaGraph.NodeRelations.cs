using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Gets the CUDA runtime local id for a graph node when supported by the loaded runtime.
    /// 在当前 CUDA runtime 支持时获取 graph node 的 local id。
    /// </summary>
    /// <param name="node">The graph node token. graph node 的 token。</param>
    /// <returns>The CUDA graph node local id. CUDA graph node 的 local id。</returns>
    public static uint GetNodeLocalId(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeLocalId(node);
    }

    /// <summary>
    /// Gets the CUDA tools id for a graph node when supported by the loaded runtime.
    /// 在当前 CUDA runtime 支持时获取 graph node 的 tools id。
    /// </summary>
    /// <param name="node">The graph node token. graph node 的 token。</param>
    /// <returns>The CUDA graph node tools id. CUDA graph node 的 tools id。</returns>
    public static ulong GetNodeToolsId(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeToolsId(node);
    }

    /// <summary>
    /// Gets the number of dependencies for a graph node.
    /// 获取指定 graph node 的依赖数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependency count. 依赖数量。</returns>
    public static ulong GetDependencyCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependencyCount(node);
    }

    /// <summary>
    /// Gets a dependency node by index for a graph node.
    /// 按索引获取指定 graph node 的依赖节点。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependency index. 从零开始的依赖索引。</param>
    /// <returns>The dependency node token. 依赖节点 token。</returns>
    public static CudaGraphNode GetDependency(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependency(node, index);
    }

    /// <summary>
    /// Gets the number of dependencies with CUDA edge data for a graph node.
    /// 获取指定 graph node 带 CUDA edge data 的依赖数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependency count. 依赖数量。</returns>
    public static ulong GetDependencyWithEdgeDataCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependencyV2Count(node);
    }

    /// <summary>
    /// Gets a dependency node and CUDA edge data by index for a graph node.
    /// 按索引获取指定 graph node 的依赖节点和 CUDA edge data。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependency index. 从零开始的依赖索引。</param>
    /// <returns>The dependency node and edge data. 依赖节点和 edge data。</returns>
    public static CudaGraphNodeDependency GetDependencyWithEdgeData(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependencyV2(node, index);
    }

    /// <summary>
    /// Gets the number of dependent nodes for a graph node.
    /// 获取依赖指定 graph node 的节点数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependent count. dependent 节点数量。</returns>
    public static ulong GetDependentCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependentCount(node);
    }

    /// <summary>
    /// Gets a dependent node by index for a graph node.
    /// 按索引获取依赖指定 graph node 的节点。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependent index. 从零开始的 dependent 节点索引。</param>
    /// <returns>The dependent node token. dependent 节点 token。</returns>
    public static CudaGraphNode GetDependent(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependent(node, index);
    }

    /// <summary>
    /// Gets the number of dependent nodes with CUDA edge data for a graph node.
    /// 获取依赖指定 graph node 且带 CUDA edge data 的节点数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependent node count. dependent 节点数量。</returns>
    public static ulong GetDependentWithEdgeDataCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependentV2Count(node);
    }

    /// <summary>
    /// Gets a dependent node and CUDA edge data by index for a graph node.
    /// 按索引获取依赖指定 graph node 的节点和 CUDA edge data。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependent index. 从零开始的 dependent 节点索引。</param>
    /// <returns>The dependent node and edge data. dependent 节点和 edge data。</returns>
    public static CudaGraphNodeDependency GetDependentWithEdgeData(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependentV2(node, index);
    }

}
