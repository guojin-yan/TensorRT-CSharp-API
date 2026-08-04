namespace JYPPX.TensorRtSharp;

/// <summary>
/// Handles copied GPU allocator metadata and approves or rejects allocation/reallocation requests.
/// 处理复制后的 GPU allocator 元数据，并批准或拒绝分配、重分配请求。
/// </summary>
/// <remarks>
/// The return value is observational for release callbacks: native code always attempts to release owner-managed memory.
/// 对释放回调，返回值仅用于观察；native 代码始终尝试释放 owner 管理的显存，避免托管异常造成泄漏。
/// </remarks>
/// <param name="request">The copied pointer-free request. 复制后的无指针请求。</param>
/// <returns><c>true</c> to permit allocation or reallocation. 返回 true 允许分配或重分配。</returns>
public delegate bool TensorRtGpuAllocatorHandler(TensorRtGpuAllocatorCallbackRequest request);
