using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns the lifetime boundary for capture into an existing CUDA graph.
/// 管理写入已有 CUDA graph 的 capture 生命周期边界。
/// </summary>
public sealed class CudaStreamCaptureToGraphSession : IDisposable
{
    private readonly CudaStream _stream;
    private readonly CudaGraph _graph;
    private readonly object _lifecycleGate = new object();
    private bool _ended;

    internal CudaStreamCaptureToGraphSession(CudaStream stream, CudaGraph graph)
    {
        _stream = stream;
        _graph = graph;
    }

    /// <summary>
    /// Gets the graph that remains the owner after the session ends.
    /// 获取 session 结束后继续拥有资源的 graph。
    /// </summary>
    public CudaGraph Graph => _graph;

    /// <summary>
    /// Gets whether End or Dispose has closed the capture session.
    /// 获取 End 或 Dispose 是否已经关闭 capture session。
    /// </summary>
    public bool IsEnded
    {
        get
        {
            lock (_lifecycleGate)
            {
                return _ended;
            }
        }
    }

    /// <summary>
    /// Ends capture and verifies that CUDA returned the same graph handle.
    /// 结束 capture，并验证 CUDA 返回的仍是同一个 graph 句柄。
    /// </summary>
    public void End()
    {
        lock (_lifecycleGate)
        {
            if (_ended)
            {
                return;
            }

            try
            {
                NativeCaptureEnd();
            }
            finally
            {
                _ended = true;
                _graph.ExitCaptureToGraphSession();
                _stream.ExitCaptureToGraphSession();
            }
        }
    }

    /// <summary>
    /// Ends an active capture session before releasing its owner references.
    /// 在释放 owner 引用前结束 active capture session。
    /// </summary>
    public void Dispose()
    {
        End();
        GC.SuppressFinalize(this);
    }

    private void NativeCaptureEnd()
    {
        Internal.Interop.NativeCudaApi.EndStreamCaptureIntoGraph(_stream.Handle, _graph.Handle);
    }
}
