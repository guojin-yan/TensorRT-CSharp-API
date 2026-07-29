using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one managed allocator dry-run diagnostic request.
/// 描述一次托管 allocator dry-run 诊断请求。
/// </summary>
/// <remarks>
/// This value is diagnostic-only. It does not carry a TensorRT device pointer, CUDA stream, or native allocator ownership.
/// 该值仅用于诊断；它不携带 TensorRT device pointer、CUDA stream 或 native allocator 所有权。
/// </remarks>
public readonly struct TensorRtAllocatorDryRunRequest
{
    /// <summary>
    /// Creates an allocator dry-run request.
    /// 创建 allocator dry-run 请求。
    /// </summary>
    /// <param name="size">The requested allocation size in bytes. 请求分配的字节数。</param>
    /// <param name="alignment">The requested alignment in bytes. 请求的字节对齐。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    public TensorRtAllocatorDryRunRequest(ulong size, ulong alignment, string reason = "")
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), "Allocator dry-run alignment must be greater than zero.");
        }

        Size = size;
        Alignment = alignment;
        Reason = reason ?? string.Empty;
    }

    /// <summary>
    /// Gets the requested allocation size in bytes.
    /// 获取请求分配的字节数。
    /// </summary>
    public ulong Size { get; }

    /// <summary>
    /// Gets the requested alignment in bytes.
    /// 获取请求的字节对齐。
    /// </summary>
    public ulong Alignment { get; }

    /// <summary>
    /// Gets the caller-provided diagnostic reason.
    /// 获取调用方提供的诊断原因。
    /// </summary>
    public string Reason { get; }

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Reason}:{Size}:{Alignment}";
    }
}
