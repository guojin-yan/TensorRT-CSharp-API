using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes copied metadata for one TensorRT output allocator callback.
/// 描述一次 TensorRT output allocator 回调复制出的元数据。
/// </summary>
/// <remarks>
/// This request contains copied metadata only. It does not carry a TensorRT output buffer, CUDA stream handle, or device
/// pointer ownership. Native code owns every CUDA allocation returned to TensorRT.
/// 该请求只包含复制出的元数据；不携带 TensorRT output buffer、CUDA stream handle 或 device pointer 所有权。
/// </remarks>
public readonly struct TensorRtOutputAllocatorCallbackRequest
{
    private const int MaxShapeRank = 8;
    private readonly long[] _shapeDimensions;

    /// <summary>
    /// Creates an output allocator owner diagnostic request.
    /// 创建 output allocator owner 诊断请求。
    /// </summary>
    /// <param name="tensorName">The copied output tensor name. 复制出的输出 tensor 名称。</param>
    /// <param name="requestedSize">The requested output buffer size in bytes. 请求的输出缓冲区字节数。</param>
    /// <param name="alignment">The requested output buffer alignment in bytes. 请求的输出缓冲区字节对齐。</param>
    /// <param name="shapeDimensions">The copied output shape dimensions. 复制出的输出 shape 维度。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    /// <param name="hasCurrentMemory">Whether TensorRT reported an existing current memory pointer. TensorRT 是否报告已有 current memory pointer。</param>
    public TensorRtOutputAllocatorCallbackRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[]? shapeDimensions,
        string reason = "",
        bool hasCurrentMemory = false)
        : this(
            TensorRtOutputAllocatorCallbackKind.ReallocateOutput,
            tensorName,
            requestedSize,
            alignment,
            shapeDimensions,
            reason,
            hasCurrentMemory,
            hasStream: false)
    {
    }

    /// <summary>
    /// Creates a copied output allocator callback request.
    /// 创建复制后的 output allocator callback 请求。
    /// </summary>
    /// <param name="kind">The callback operation. callback 操作。</param>
    /// <param name="tensorName">The copied output tensor name. 复制出的输出 tensor 名称。</param>
    /// <param name="requestedSize">The requested output size, or zero for shape notification. 请求的输出大小；shape 通知时为零。</param>
    /// <param name="alignment">The requested alignment, or zero for shape notification. 请求的对齐；shape 通知时为零。</param>
    /// <param name="shapeDimensions">The copied shape dimensions. 复制出的 shape 维度。</param>
    /// <param name="reason">A copied diagnostic reason. 复制出的诊断原因。</param>
    /// <param name="hasCurrentMemory">Whether TensorRT supplied current memory. TensorRT 是否提供 current memory。</param>
    /// <param name="hasStream">Whether the async callback supplied a CUDA stream. 异步 callback 是否提供 CUDA stream。</param>
    public TensorRtOutputAllocatorCallbackRequest(
        TensorRtOutputAllocatorCallbackKind kind,
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[]? shapeDimensions,
        string reason = "",
        bool hasCurrentMemory = false,
        bool hasStream = false)
    {
        if (kind != TensorRtOutputAllocatorCallbackKind.NotifyShape &&
            kind != TensorRtOutputAllocatorCallbackKind.ReallocateOutput)
        {
            throw new ArgumentOutOfRangeException(nameof(kind));
        }

        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Output allocator tensor name must not be empty.", nameof(tensorName));
        }

        if (kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput && alignment == 0UL)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), "Output allocator alignment must be greater than zero.");
        }

        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        if (_shapeDimensions.Length > MaxShapeRank)
        {
            throw new ArgumentOutOfRangeException(nameof(shapeDimensions), "Output allocator diagnostic shape rank must be 8 or less.");
        }

        Kind = kind;
        TensorName = tensorName;
        RequestedSize = requestedSize;
        Alignment = alignment;
        Reason = reason ?? string.Empty;
        HasCurrentMemory = hasCurrentMemory;
        HasStream = hasStream;
    }

    /// <summary>Gets the callback operation. 获取 callback 操作。</summary>
    public TensorRtOutputAllocatorCallbackKind Kind { get; }

    /// <summary>Gets the copied output tensor name. 获取复制出的输出 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the requested output buffer size in bytes. 获取请求的输出缓冲区字节数。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets the requested output buffer alignment in bytes. 获取请求的输出缓冲区字节对齐。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets the copied output shape rank. 获取复制出的输出 shape rank。</summary>
    public int ShapeRank => _shapeDimensions.Length;

    /// <summary>Gets the copied output shape dimensions. 获取复制出的输出 shape 维度。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions ?? Array.Empty<long>());

    /// <summary>Gets the copied diagnostic reason. 获取复制出的诊断原因。</summary>
    public string Reason { get; }

    /// <summary>Gets whether an existing current memory pointer was reported. 获取是否报告了已有 current memory pointer。</summary>
    public bool HasCurrentMemory { get; }

    /// <summary>Gets whether TensorRT supplied an asynchronous CUDA stream. 获取 TensorRT 是否提供异步 CUDA stream。</summary>
    public bool HasStream { get; }

    internal long[] CopyShapeDimensions()
    {
        return _shapeDimensions == null ? Array.Empty<long>() : (long[])_shapeDimensions.Clone();
    }
}
