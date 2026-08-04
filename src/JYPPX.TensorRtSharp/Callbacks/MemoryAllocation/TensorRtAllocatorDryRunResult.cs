using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports the result of an allocator dry-run diagnostic.
/// 表示 allocator dry-run 诊断结果。
/// </summary>
/// <remarks>
/// The result intentionally contains no pointer value. It is a managed readiness signal, not an allocation result.
/// 该结果故意不包含任何指针值；它是托管可用性信号，不是分配结果。
/// </remarks>
public readonly struct TensorRtAllocatorDryRunResult
{
    /// <summary>
    /// Creates a dry-run result.
    /// 创建 dry-run 结果。
    /// </summary>
    /// <param name="succeeded">Whether the diagnostic completed successfully. 诊断是否成功完成。</param>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    public TensorRtAllocatorDryRunResult(bool succeeded, string diagnostic)
    {
        Succeeded = succeeded;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets whether the diagnostic completed successfully.
    /// 获取诊断是否成功完成。
    /// </summary>
    public bool Succeeded { get; }

    /// <summary>
    /// Gets a copied diagnostic message.
    /// 获取复制出的诊断消息。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Creates a successful dry-run result.
    /// 创建成功的 dry-run 结果。
    /// </summary>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    /// <returns>A successful result. 成功结果。</returns>
    public static TensorRtAllocatorDryRunResult Success(string diagnostic = "OK")
    {
        return new TensorRtAllocatorDryRunResult(true, diagnostic);
    }

    /// <summary>
    /// Creates a failed dry-run result.
    /// 创建失败的 dry-run 结果。
    /// </summary>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    /// <returns>A failed result. 失败结果。</returns>
    public static TensorRtAllocatorDryRunResult Failure(string diagnostic)
    {
        return new TensorRtAllocatorDryRunResult(false, diagnostic);
    }

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Succeeded}:{Diagnostic}";
    }
}
