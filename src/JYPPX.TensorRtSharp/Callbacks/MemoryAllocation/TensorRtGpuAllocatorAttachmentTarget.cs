namespace JYPPX.TensorRtSharp;

/// <summary>Identifies the TensorRT object currently borrowing a GPU allocator owner. 标识当前借用 GPU allocator owner 的 TensorRT 对象。</summary>
public enum TensorRtGpuAllocatorAttachmentTarget
{
    /// <summary>The owner is not attached. owner 未挂载。</summary>
    None = 0,

    /// <summary>The owner is attached to a runtime. owner 挂载到 runtime。</summary>
    Runtime = 1,

    /// <summary>The owner is attached to a builder. owner 挂载到 builder。</summary>
    Builder = 2
}
