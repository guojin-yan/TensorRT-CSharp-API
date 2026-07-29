using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaStream
{
    /// <summary>
    /// Begins CUDA graph capture on this stream.
    /// 在当前 stream 上开始 CUDA graph capture。
    /// </summary>
    /// <param name="mode">The capture-validation mode. capture 校验模式。</param>
    public void BeginCapture(CudaStreamCaptureMode mode = CudaStreamCaptureMode.Global)
    {
        NativeCudaApi.BeginStreamCapture(_handle, mode);
    }

    /// <summary>
    /// Begins capture into an existing graph and returns an owner-scoped session.
    /// 将 capture 写入已有 graph，并返回 owner-scoped session。
    /// </summary>
    /// <param name="graph">The graph that remains the owner after End. End 后继续拥有该 graph 的对象。</param>
    /// <param name="dependencies">Graph-owned dependency nodes and copied edge data. graph-owned dependency 节点及复制型 edge data。</param>
    /// <param name="mode">The capture-validation mode. capture 校验模式。</param>
    /// <returns>A session that must be ended before the stream or graph is disposed. 必须在 stream 或 graph dispose 前结束的 session。</returns>
    public CudaStreamCaptureToGraphSession BeginCaptureToGraph(
        CudaGraph graph,
        IReadOnlyList<CudaGraphNodeDependency> dependencies,
        CudaStreamCaptureMode mode = CudaStreamCaptureMode.Global)
    {
        if (graph == null)
        {
            throw new ArgumentNullException(nameof(graph));
        }
        if (dependencies == null)
        {
            throw new ArgumentNullException(nameof(dependencies));
        }
        if (mode != CudaStreamCaptureMode.Global &&
            mode != CudaStreamCaptureMode.ThreadLocal &&
            mode != CudaStreamCaptureMode.Relaxed)
        {
            throw new ArgumentOutOfRangeException(nameof(mode));
        }

        EnterCaptureToGraphSession();
        try
        {
            graph.EnterCaptureToGraphSession();
            try
            {
                NativeCudaApi.BeginStreamCaptureToGraph(_handle, graph.Handle, dependencies, mode);
                return new CudaStreamCaptureToGraphSession(this, graph);
            }
            catch
            {
                graph.ExitCaptureToGraphSession();
                throw;
            }
        }
        catch
        {
            ExitCaptureToGraphSession();
            throw;
        }
    }

    /// <summary>
    /// Ends CUDA graph capture and returns the captured graph.
    /// 结束 CUDA graph capture 并返回捕获得到的 graph。
    /// </summary>
    /// <returns>The captured CUDA graph. 捕获得到的 CUDA graph。</returns>
    public CudaGraph EndCapture()
    {
        return new CudaGraph(NativeCudaApi.EndStreamCapture(_handle));
    }

}
