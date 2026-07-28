namespace JYPPX.CudaSharp;

/// <summary>
/// Describes copied, pointer-free CUDA graph node parameters.
/// 描述复制出的、无指针暴露的 CUDA graph node 参数。
/// </summary>
/// <remarks>
/// This descriptor intentionally excludes borrowed CUDA handles and native addresses. It is safe to expose on the public API surface.
/// 该 descriptor 有意排除 borrowed CUDA handle 和原生地址，适合作为 public API 暴露。
/// </remarks>
public readonly struct CudaGraphNodeParamsDescriptor
{
    private CudaGraphNodeParamsDescriptor(
        CudaGraphNodeType nodeType,
        CudaGraphNodeParamsDescriptorKind descriptorKind,
        bool hasCopiedParameters,
        bool hasBorrowedHandleExposure,
        ulong byteCount,
        ulong height,
        ulong depth,
        uint value,
        uint elementSize,
        CudaMemcpyKind memcpyKind,
        bool hasEvent,
        string status)
    {
        NodeType = nodeType;
        DescriptorKind = descriptorKind;
        HasCopiedParameters = hasCopiedParameters;
        HasBorrowedHandleExposure = hasBorrowedHandleExposure;
        ByteCount = byteCount;
        Height = height;
        Depth = depth;
        Value = value;
        ElementSize = elementSize;
        MemcpyKind = memcpyKind;
        HasEvent = hasEvent;
        Status = status;
    }

    /// <summary>
    /// Gets the CUDA graph node type.
    /// 获取 CUDA graph node 类型。
    /// </summary>
    public CudaGraphNodeType NodeType { get; }

    /// <summary>
    /// Gets the descriptor kind used by this copied summary.
    /// 获取该复制型摘要使用的 descriptor 类型。
    /// </summary>
    public CudaGraphNodeParamsDescriptorKind DescriptorKind { get; }

    /// <summary>
    /// Gets whether this descriptor includes copied node parameters.
    /// 获取该 descriptor 是否包含复制出的节点参数。
    /// </summary>
    public bool HasCopiedParameters { get; }

    /// <summary>
    /// Gets whether this descriptor exposes borrowed handles or native pointers. This must remain false.
    /// 获取该 descriptor 是否暴露 borrowed handle 或原生指针；该值必须保持 false。
    /// </summary>
    public bool HasBorrowedHandleExposure { get; }

    /// <summary>
    /// Gets the copied byte count for 1D memcpy/memset descriptors when available.
    /// 在可用时获取 1D memcpy/memset descriptor 的复制字节数。
    /// </summary>
    public ulong ByteCount { get; }

    /// <summary>
    /// Gets the copied height for node parameters when available.
    /// 在可用时获取节点参数复制出的高度。
    /// </summary>
    public ulong Height { get; }

    /// <summary>
    /// Gets the copied depth for node parameters when available.
    /// 在可用时获取节点参数复制出的深度。
    /// </summary>
    public ulong Depth { get; }

    /// <summary>
    /// Gets the copied memset value when this is a memset descriptor.
    /// 在 memset descriptor 中获取复制出的填充值。
    /// </summary>
    public uint Value { get; }

    /// <summary>
    /// Gets the copied memset element size when this is a memset descriptor.
    /// 在 memset descriptor 中获取复制出的元素大小。
    /// </summary>
    public uint ElementSize { get; }

    /// <summary>
    /// Gets the copied memcpy direction when this is a memcpy descriptor.
    /// 在 memcpy descriptor 中获取复制出的 memcpy 方向。
    /// </summary>
    public CudaMemcpyKind MemcpyKind { get; }

    /// <summary>
    /// Gets whether an event node has a caller-owned event without exposing the borrowed event handle.
    /// 获取 event node 是否引用调用方拥有的 event，但不暴露 borrowed event handle。
    /// </summary>
    public bool HasEvent { get; }

    /// <summary>
    /// Gets the descriptor status text.
    /// 获取 descriptor 状态文本。
    /// </summary>
    public string Status { get; }

    /// <summary>
    /// Creates a pointer-free descriptor for an empty node.
    /// 为 empty node 创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor Empty() =>
        new(
            CudaGraphNodeType.Empty,
            CudaGraphNodeParamsDescriptorKind.Empty,
            true,
            false,
            0,
            0,
            0,
            0,
            0,
            default,
            false,
            "copied-empty-node");

    /// <summary>
    /// Creates a pointer-free descriptor for an event-record node.
    /// 为 event-record node 创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor EventRecord(bool hasEvent) =>
        new(
            CudaGraphNodeType.EventRecord,
            CudaGraphNodeParamsDescriptorKind.EventRecord,
            true,
            false,
            0,
            0,
            0,
            0,
            0,
            default,
            hasEvent,
            "copied-event-presence-no-borrowed-handle");

    /// <summary>
    /// Creates a pointer-free descriptor for an event-wait node.
    /// 为 event-wait node 创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor EventWait(bool hasEvent) =>
        new(
            CudaGraphNodeType.WaitEvent,
            CudaGraphNodeParamsDescriptorKind.EventWait,
            true,
            false,
            0,
            0,
            0,
            0,
            0,
            default,
            hasEvent,
            "copied-event-presence-no-borrowed-handle");

    /// <summary>
    /// Creates a pointer-free descriptor for a memset node.
    /// 为 memset node 创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor Memset(CudaGraphMemsetNodeParameters parameters) =>
        new(
            CudaGraphNodeType.Memset,
            CudaGraphNodeParamsDescriptorKind.Memset,
            true,
            false,
            parameters.Width,
            parameters.Height,
            1,
            parameters.Value,
            parameters.ElementSize,
            default,
            false,
            "copied-memset-scalars-no-address");

    /// <summary>
    /// Creates a pointer-free descriptor for a memcpy node.
    /// 为 memcpy node 创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor Memcpy(CudaGraphMemcpyNodeParameters parameters) =>
        new(
            CudaGraphNodeType.Memcpy,
            CudaGraphNodeParamsDescriptorKind.Memcpy,
            true,
            false,
            parameters.Width,
            parameters.Height,
            parameters.Depth,
            0,
            0,
            parameters.Kind,
            false,
            "copied-memcpy-scalars-no-address");

    /// <summary>
    /// Creates a pointer-free descriptor for a node kind whose parameters are intentionally not exposed.
    /// 为参数尚未安全暴露的节点类型创建无指针 descriptor。
    /// </summary>
    public static CudaGraphNodeParamsDescriptor Unsupported(CudaGraphNodeType nodeType) =>
        new(
            nodeType,
            CudaGraphNodeParamsDescriptorKind.Unsupported,
            false,
            false,
            0,
            0,
            0,
            0,
            0,
            default,
            false,
            "unsupported-or-deferred-no-borrowed-handle");

    /// <summary>
    /// Formats this descriptor for diagnostics.
    /// 将该 descriptor 格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"Kind={DescriptorKind}, NodeType={NodeType}, Copied={HasCopiedParameters}, BorrowedHandleExposure={HasBorrowedHandleExposure}, Bytes={ByteCount}, Height={Height}, Depth={Depth}, MemcpyKind={MemcpyKind}, Value={Value}, ElementSize={ElementSize}, HasEvent={HasEvent}, Status={Status}";
}
