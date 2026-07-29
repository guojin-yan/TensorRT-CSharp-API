using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Instantiates this graph as an executable CUDA graph.
    /// 将当前 graph 实例化为可执行 CUDA graph。
    /// </summary>
    /// <param name="flags">The CUDA graph-instantiation flags. CUDA graph 实例化标志。</param>
    /// <returns>A managed executable graph wrapper. 可执行 graph 的托管封装。</returns>
    public CudaGraphExec Instantiate(ulong flags = 0)
    {
        return new CudaGraphExec(NativeCudaApi.InstantiateGraph(_handle, flags));
    }

    /// <summary>
    /// Instantiates this graph through CUDA's parameterized instantiation API.
    /// 通过 CUDA 参数化实例化 API 将当前 graph 实例化。
    /// </summary>
    /// <param name="flags">CUDA graph instantiation flags. CUDA graph 实例化标志。</param>
    /// <returns>A managed executable graph. 托管 executable graph。</returns>
    public CudaGraphExec InstantiateWithParameters(ulong flags = 0)
    {
        return new CudaGraphExec(NativeCudaApi.InstantiateGraphWithParameters(_handle, flags));
    }

    /// <summary>
    /// Instantiates and uploads this graph through CUDA's parameterized instantiation API.
    /// 通过 CUDA 参数化实例化 API 实例化并上传当前 graph。
    /// </summary>
    /// <param name="uploadStream">The managed stream used for upload. 用于 upload 的托管 stream。</param>
    /// <param name="flags">CUDA graph instantiation flags. CUDA graph 实例化标志。</param>
    /// <returns>A managed executable graph. 托管 executable graph。</returns>
    public CudaGraphExec InstantiateWithParameters(CudaStream uploadStream, ulong flags = 0)
    {
        if (uploadStream == null)
        {
            throw new ArgumentNullException(nameof(uploadStream));
        }

        return new CudaGraphExec(NativeCudaApi.InstantiateGraphWithParameters(_handle, flags, uploadStream.Handle));
    }

}
