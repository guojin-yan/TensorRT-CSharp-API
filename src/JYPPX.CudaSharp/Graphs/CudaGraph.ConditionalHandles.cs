using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Creates a CUDA conditional handle associated with this graph.
    /// 创建一个与当前 graph 关联的 CUDA conditional handle。
    /// </summary>
    public CudaGraphConditionalHandle CreateConditionalHandle(
        uint defaultLaunchValue = 0,
        CudaGraphConditionalHandleFlags flags = CudaGraphConditionalHandleFlags.None)
    {
        ThrowIfDisposed();
        return new CudaGraphConditionalHandle(
            this,
            NativeCudaApi.CreateConditionalHandle(_handle, defaultLaunchValue, flags),
            defaultLaunchValue,
            flags);
    }

    /// <summary>
    /// Creates a CUDA 13 conditional handle, optionally bound to a primary execution context.
    /// 创建 CUDA 13 conditional handle，可选绑定到主 execution context。
    /// </summary>
    public CudaGraphConditionalHandle CreateConditionalHandleV2(
        CudaPrimaryExecutionContext? context = null,
        uint defaultLaunchValue = 0,
        CudaGraphConditionalHandleFlags flags = CudaGraphConditionalHandleFlags.None)
    {
        ThrowIfDisposed();
        return new CudaGraphConditionalHandle(
            this,
            NativeCudaApi.CreateConditionalHandleV2(_handle, context?.Handle, defaultLaunchValue, flags),
            defaultLaunchValue,
            flags);
    }

    /// <summary>
    /// Gets the number of nodes currently in this graph.
    /// 获取当前 graph 中的 node 数量。
    /// </summary>
    public ulong NodeCount => NativeCudaApi.GetGraphNodeCount(_handle);

    /// <summary>
    /// Gets the number of root nodes currently in this graph.
    /// 获取当前 graph 中的 root node 数量。
    /// </summary>
    public ulong RootNodeCount => NativeCudaApi.GetGraphRootNodeCount(_handle);

    /// <summary>
    /// Gets the number of dependency edges currently in this graph.
    /// 获取当前 graph 中的依赖 edge 数量。
    /// </summary>
    public ulong EdgeCount => NativeCudaApi.GetGraphEdgeCount(_handle);

    /// <summary>
    /// Gets the CUDA graph identifier when supported by the loaded CUDA runtime.
    /// 在当前 CUDA runtime 支持时获取 CUDA graph 标识符。
    /// </summary>
    public uint Id => NativeCudaApi.GetGraphId(_handle);

}
