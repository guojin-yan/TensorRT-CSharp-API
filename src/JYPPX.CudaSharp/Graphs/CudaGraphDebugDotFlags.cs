using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control which additional information CUDA writes to graph debug DOT output.
/// 控制 CUDA graph debug DOT 输出附加信息的标志。
/// </summary>
[Flags]
public enum CudaGraphDebugDotFlags : uint
{
    /// <summary>
    /// Writes the default DOT output.
    /// 写出默认 DOT 输出。
    /// </summary>
    None = 0,

    /// <summary>
    /// Outputs all debug data as if every debug flag is enabled.
    /// 输出所有 debug 数据。
    /// </summary>
    Verbose = 1U << 0,

    /// <summary>
    /// Adds CUDA kernel node parameters to output.
    /// 输出 CUDA kernel node 参数。
    /// </summary>
    KernelNodeParams = 1U << 2,

    /// <summary>
    /// Adds CUDA memcpy node parameters to output.
    /// 输出 CUDA memcpy node 参数。
    /// </summary>
    MemcpyNodeParams = 1U << 3,

    /// <summary>
    /// Adds CUDA memset node parameters to output.
    /// 输出 CUDA memset node 参数。
    /// </summary>
    MemsetNodeParams = 1U << 4,

    /// <summary>
    /// Adds CUDA host node parameters to output.
    /// 输出 CUDA host node 参数。
    /// </summary>
    HostNodeParams = 1U << 5,

    /// <summary>
    /// Adds CUDA event node parameters to output.
    /// 输出 CUDA event node 参数。
    /// </summary>
    EventNodeParams = 1U << 6,

    /// <summary>
    /// Adds CUDA external semaphore signal node parameters to output.
    /// 输出 CUDA external semaphore signal node 参数。
    /// </summary>
    ExternalSemaphoreSignalNodeParams = 1U << 7,

    /// <summary>
    /// Adds CUDA external semaphore wait node parameters to output.
    /// 输出 CUDA external semaphore wait node 参数。
    /// </summary>
    ExternalSemaphoreWaitNodeParams = 1U << 8,

    /// <summary>
    /// Adds CUDA kernel node attributes to output.
    /// 输出 CUDA kernel node attribute。
    /// </summary>
    KernelNodeAttributes = 1U << 9,

    /// <summary>
    /// Adds node handles and kernel function handles to output.
    /// 输出 node handle 和 kernel function handle。
    /// </summary>
    Handles = 1U << 10,

    /// <summary>
    /// Adds CUDA conditional node parameters to output.
    /// 输出 CUDA conditional node 参数。
    /// </summary>
    ConditionalNodeParams = 1U << 15
}
