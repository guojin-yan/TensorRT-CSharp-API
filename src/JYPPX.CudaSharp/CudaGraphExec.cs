using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around an executable CUDA graph.
/// </summary>
public sealed class CudaGraphExec : IDisposable
{
    private readonly SafeCudaGraphExecHandle _handle;

    internal CudaGraphExec(SafeCudaGraphExecHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    internal SafeCudaGraphExecHandle Handle => _handle;

    /// <summary>
    /// Gets graph executable flags when supported by the CUDA runtime.
    /// 在 CUDA runtime 支持时获取 graph executable flags。
    /// </summary>
    public ulong Flags => NativeCudaApi.GetGraphExecFlags(_handle);

    /// <summary>
    /// Gets the CUDA graph executable identifier when supported by the loaded CUDA runtime.
    /// 在当前 CUDA runtime 支持时获取 CUDA graph executable 标识符。
    /// </summary>
    public uint Id => NativeCudaApi.GetGraphExecId(_handle);

    /// <summary>
    /// Sets whether a node is enabled in this executable graph.
    /// 设置 executable graph 中某个节点是否启用。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <param name="enabled">Whether the node should be enabled. 节点是否启用。</param>
    public void SetNodeEnabled(CudaGraphNode node, bool enabled)
    {
        NativeCudaApi.SetGraphExecNodeEnabled(_handle, node, enabled);
    }

    /// <summary>
    /// Gets whether a node is enabled in this executable graph.
    /// 获取 executable graph 中某个节点是否启用。
    /// </summary>
    /// <param name="node">A node token owned by the source graph. 源 graph 拥有的节点 token。</param>
    /// <returns>True when the node is enabled. 节点启用时返回 true。</returns>
    public bool GetNodeEnabled(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphExecNodeEnabled(_handle, node);
    }

    /// <summary>
    /// Uploads this executable graph to a CUDA stream before launch.
    /// 在 launch 前将当前可执行 graph 上传到 CUDA stream。
    /// </summary>
    /// <param name="stream">The stream used for graph upload. 用于 graph upload 的 stream。</param>
    public void Upload(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.UploadGraphExec(_handle, stream.Handle);
    }

    public void Launch(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.LaunchGraphExec(_handle, stream.Handle);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
