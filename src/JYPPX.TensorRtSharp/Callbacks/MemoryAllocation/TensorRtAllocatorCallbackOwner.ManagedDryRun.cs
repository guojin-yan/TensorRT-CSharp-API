using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
    /// <summary>
    /// Runs the managed allocator dry-run diagnostic.
    /// 执行托管 allocator dry-run 诊断。
    /// </summary>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <returns>A copied diagnostic result that never contains a device pointer. 不包含 device pointer 的诊断结果副本。</returns>
    /// <remarks>
    /// This method does not call TensorRT, CUDA, or a native allocator trampoline. It is intended to validate managed
    /// callback lifetime, exception capture, and package-consumer surface before real allocator callbacks are unlocked.
    /// 该方法不会调用 TensorRT、CUDA 或 native allocator trampoline。它用于在解锁真实 allocator callback 前验证托管
    /// 回调生命周期、异常捕获与 package-consumer API 表面。
    /// </remarks>
    public TensorRtAllocatorDryRunResult RunDryRunDiagnostic(TensorRtAllocatorDryRunRequest request)
    {
        ThrowIfDisposed();
        _callbackState.RecordInvocation();

        try
        {
            TensorRtAllocatorDryRunResult result = _callbackState.Handler(request);
            string diagnostic = string.IsNullOrWhiteSpace(result.Diagnostic)
                ? (result.Succeeded ? "OK" : "allocator dry-run handler returned failure without a diagnostic.")
                : result.Diagnostic;

            result = new TensorRtAllocatorDryRunResult(result.Succeeded, diagnostic);
            _callbackState.RecordDiagnostic(result.Diagnostic);
            return result;
        }
        catch (Exception exception)
        {
            string diagnostic = "allocator dry-run handler threw " + exception.GetType().Name + ": " + exception.Message;
            _callbackState.RecordFailure(exception, diagnostic);
            return TensorRtAllocatorDryRunResult.Failure(diagnostic);
        }
    }

}
