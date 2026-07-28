namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies the copied, pointer-free CUDA graph node parameter descriptor kind.
/// 标识复制型、无指针 CUDA graph node 参数 descriptor 类型。
/// </summary>
public enum CudaGraphNodeParamsDescriptorKind
{
    /// <summary>
    /// The node kind is not yet represented by a safe typed descriptor.
    /// 该节点类型尚未由安全 typed descriptor 表示。
    /// </summary>
    Unsupported = 0,

    /// <summary>
    /// Empty node descriptor.
    /// 空节点描述符。
    /// </summary>
    Empty = 1,

    /// <summary>
    /// Memset node descriptor.
    /// Memset 节点描述符。
    /// </summary>
    Memset = 2,

    /// <summary>
    /// Memcpy node descriptor.
    /// Memcpy 节点描述符。
    /// </summary>
    Memcpy = 3,

    /// <summary>
    /// Event-record node descriptor.
    /// 事件记录节点描述符。
    /// </summary>
    EventRecord = 4,

    /// <summary>
    /// Event-wait node descriptor.
    /// 事件等待节点描述符。
    /// </summary>
    EventWait = 5
}
