using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies CUDA graph node kinds reported by the CUDA runtime.
/// 标识 CUDA runtime 报告的 CUDA graph node 类型。
/// </summary>
public enum CudaGraphNodeType
{
    /// <summary>
    /// A kernel-launch node.
    /// kernel 启动节点。
    /// </summary>
    Kernel = 0,
    /// <summary>
    /// A memory-copy node.
    /// 内存复制节点。
    /// </summary>
    Memcpy = 1,
    /// <summary>
    /// A memory-set node.
    /// 内存填充节点。
    /// </summary>
    Memset = 2,
    /// <summary>
    /// A host-callback node.
    /// 主机回调节点。
    /// </summary>
    Host = 3,
    /// <summary>
    /// A child-graph node.
    /// 子 graph 节点。
    /// </summary>
    Graph = 4,
    /// <summary>
    /// An empty synchronization node.
    /// 空同步节点。
    /// </summary>
    Empty = 5,
    /// <summary>
    /// An event-wait node.
    /// event 等待节点。
    /// </summary>
    WaitEvent = 6,
    /// <summary>
    /// An event-record node.
    /// event 记录节点。
    /// </summary>
    EventRecord = 7,
    /// <summary>
    /// An external-semaphore signal node.
    /// 外部 semaphore signal 节点。
    /// </summary>
    ExternalSemaphoreSignal = 8,
    /// <summary>
    /// An external-semaphore wait node.
    /// 外部 semaphore wait 节点。
    /// </summary>
    ExternalSemaphoreWait = 9,
    /// <summary>
    /// A memory-allocation node.
    /// 内存分配节点。
    /// </summary>
    MemoryAlloc = 10,
    /// <summary>
    /// A memory-free node.
    /// 内存释放节点。
    /// </summary>
    MemoryFree = 11,
    /// <summary>
    /// A batch memory-operation node.
    /// 批量内存操作节点。
    /// </summary>
    BatchMemoryOperation = 12,
    /// <summary>
    /// A conditional-execution node.
    /// 条件执行节点。
    /// </summary>
    Conditional = 13
}
