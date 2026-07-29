using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control CUDA event creation behavior.
/// 控制 CUDA event 创建行为的标志。
/// </summary>
[Flags]
public enum CudaEventCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default event-creation behavior.
    /// 使用 CUDA 默认的 event 创建行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Creates an event that blocks the waiting host thread.
    /// 创建一个会阻塞等待线程的 event。
    /// </summary>
    BlockingSync = 1,
    /// <summary>
    /// Creates an event without timing information.
    /// 创建一个不记录计时信息的 event。
    /// </summary>
    DisableTiming = 2,
    /// <summary>
    /// Creates an event that can be shared across processes.
    /// 创建一个可跨进程共享的 event。
    /// </summary>
    Interprocess = 4
}
/// <summary>
/// Flags used when recording an event into a stream.
/// 向 stream 记录 event 时使用的标志。
/// </summary>
[Flags]
public enum CudaEventRecordFlags : uint
{
    /// <summary>
    /// Use CUDA's default event-record behavior.
    /// 使用 CUDA 默认 event record 行为。
    /// </summary>
    Default = 0,

    /// <summary>
    /// Capture the event as an external event node during graph capture.
    /// graph capture 期间将 event 捕获为 external event node。
    /// </summary>
    External = 1
}
