namespace JYPPX.TensorRtSharp;

/// <summary>
/// Contains copied, pointer-free metadata for one TensorRT GPU allocator callback.
/// 包含一次 TensorRT GPU allocator 回调复制出的无指针元数据。
/// </summary>
public readonly struct TensorRtGpuAllocatorCallbackRequest
{
    internal TensorRtGpuAllocatorCallbackRequest(
        TensorRtGpuAllocatorCallbackKind kind,
        ulong requestedSize,
        ulong alignment,
        uint allocatorFlags,
        bool hasCurrentMemory,
        bool hasStream)
    {
        Kind = kind;
        RequestedSize = requestedSize;
        Alignment = alignment;
        AllocatorFlags = allocatorFlags;
        HasCurrentMemory = hasCurrentMemory;
        HasStream = hasStream;
    }

    /// <summary>Gets the callback operation. 获取回调操作。</summary>
    public TensorRtGpuAllocatorCallbackKind Kind { get; }

    /// <summary>Gets requested bytes; release callbacks report zero. 获取请求字节数；释放回调为零。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets requested alignment; release callbacks report zero. 获取请求对齐；释放回调为零。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets TensorRT allocator flags copied as an unsigned mask. 获取以无符号掩码复制的 TensorRT allocator flags。</summary>
    public uint AllocatorFlags { get; }

    /// <summary>Gets whether TensorRT supplied existing memory without exposing its address. 获取 TensorRT 是否提供已有显存，但不暴露其地址。</summary>
    public bool HasCurrentMemory { get; }

    /// <summary>Gets whether TensorRT supplied a CUDA stream without exposing its handle. 获取 TensorRT 是否提供 CUDA stream，但不暴露其句柄。</summary>
    public bool HasStream { get; }

    /// <summary>Gets whether this is a release notification. 获取当前操作是否为释放通知。</summary>
    public bool IsRelease => Kind == TensorRtGpuAllocatorCallbackKind.Deallocate || Kind == TensorRtGpuAllocatorCallbackKind.DeallocateAsync;

    /// <summary>Gets whether this API exposes a native pointer. 获取该 API 是否暴露 native pointer。</summary>
    public bool NativePointerExposed => false;

    /// <summary>Returns a compact diagnostic string. 返回紧凑诊断字符串。</summary>
    public override string ToString() => $"{Kind}:size={RequestedSize}:alignment={Alignment}:flags={AllocatorFlags}:current={HasCurrentMemory}:stream={HasStream}";
}
