namespace JYPPX.TensorRtSharp;

/// <summary>
/// Identifies an <c>IOutputAllocator</c> callback operation.
/// 标识一次 <c>IOutputAllocator</c> 回调操作。
/// </summary>
public enum TensorRtOutputAllocatorCallbackKind
{
    /// <summary>An unknown callback operation. 未知回调操作。</summary>
    Unknown = 0,

    /// <summary>TensorRT reported the resolved output shape. TensorRT 报告已解析的输出 shape。</summary>
    NotifyShape = 1,

    /// <summary>TensorRT requested output memory. TensorRT 请求输出内存。</summary>
    ReallocateOutput = 2
}
