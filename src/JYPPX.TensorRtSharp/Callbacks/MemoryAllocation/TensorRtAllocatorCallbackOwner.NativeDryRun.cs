using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
    /// <summary>
    /// Runs the native allocator owner dry-run C ABI diagnostic.
    /// 执行 native allocator owner dry-run C ABI 诊断。
    /// </summary>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <returns>Copied native dry-run diagnostics without any native handle or device pointer. 不包含 native handle 或 device pointer 的诊断副本。</returns>
    /// <remarks>
    /// This method creates a short-lived native diagnostic owner, emits one dry-run diagnostic, copies counters/status
    /// back to managed memory, and immediately releases the native handle. It does not attach to TensorRT, does not call
    /// <c>setGpuAllocator</c>, and does not implement <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>.
    /// 该方法会创建一个短生命周期 native 诊断 owner，发出一次 dry-run 诊断，将计数与状态复制回托管内存，然后立即释放
    /// native 句柄。它不会绑定 TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现
    /// <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>。
    /// </remarks>
    public TensorRtAllocatorNativeDryRunResult RunNativeDryRunDiagnostic(TensorRtApiLine line, TensorRtAllocatorDryRunRequest request)
    {
        ThrowIfDisposed();

        using SafeTensorRtObjectHandle nativeOwner = NativeBridgeApi.CreateAllocatorOwnerDryRun(line);
        NativeTensorRtAllocatorOwnerDiagnosticInfo nativeInfo =
            NativeBridgeApi.EmitAllocatorOwnerDryRunDiagnostic(line, nativeOwner, request.Size, request.Alignment, request.Reason);
        return CreateNativeDryRunResult(nativeInfo);
    }

}
