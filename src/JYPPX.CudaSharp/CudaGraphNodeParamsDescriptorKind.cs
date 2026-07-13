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
    /// Empty node descriptor。
    /// </summary>
    Empty = 1,

    /// <summary>
    /// Memset node descriptor.
    /// Memset node descriptor。
    /// </summary>
    Memset = 2,

    /// <summary>
    /// Memcpy node descriptor.
    /// Memcpy node descriptor。
    /// </summary>
    Memcpy = 3,

    /// <summary>
    /// Event-record node descriptor.
    /// Event-record node descriptor。
    /// </summary>
    EventRecord = 4,

    /// <summary>
    /// Event-wait node descriptor.
    /// Event-wait node descriptor。
    /// </summary>
    EventWait = 5
}
