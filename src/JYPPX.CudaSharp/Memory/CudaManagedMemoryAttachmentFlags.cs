using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control CUDA managed-memory attachment behavior.
/// 控制 CUDA managed memory 附着行为的标志。
/// </summary>
[Flags]
public enum CudaManagedMemoryAttachmentFlags : uint
{
    /// <summary>
    /// Attaches the allocation globally.
    /// 全局附着该分配。
    /// </summary>
    Global = 1,
    /// <summary>
    /// Attaches the allocation to the host.
    /// 将该分配附着到 host。
    /// </summary>
    Host = 2,
    /// <summary>
    /// Attaches the allocation to a single stream.
    /// 将该分配附着到单个 stream。
    /// </summary>
    Single = 4
}
