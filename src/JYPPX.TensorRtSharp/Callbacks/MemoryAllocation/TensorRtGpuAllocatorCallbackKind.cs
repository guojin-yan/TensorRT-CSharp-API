namespace JYPPX.TensorRtSharp;

/// <summary>Identifies an <c>IGpuAllocator</c> callback operation. 标识 IGpuAllocator 回调操作。</summary>
public enum TensorRtGpuAllocatorCallbackKind
{
    /// <summary>No recognized operation. 未识别操作。</summary>
    Unknown = 0,

    /// <summary>Synchronous allocation. 同步分配。</summary>
    Allocate = 1,

    /// <summary>Reallocation of owner-managed memory. 重分配 owner 管理的显存。</summary>
    Reallocate = 2,

    /// <summary>Synchronous release. 同步释放。</summary>
    Deallocate = 3,

    /// <summary>Stream-aware allocation request. 带 CUDA stream 的分配请求。</summary>
    AllocateAsync = 4,

    /// <summary>Stream-aware release request. 带 CUDA stream 的释放请求。</summary>
    DeallocateAsync = 5
}
