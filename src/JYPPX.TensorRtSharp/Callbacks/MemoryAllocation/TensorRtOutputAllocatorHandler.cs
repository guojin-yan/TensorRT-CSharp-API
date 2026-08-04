namespace JYPPX.TensorRtSharp;

/// <summary>
/// Handles copied metadata from a native TensorRT output allocator.
/// 处理 native TensorRT output allocator 复制出的元数据。
/// </summary>
/// <param name="request">The pointer-free callback request. 无指针 callback 请求。</param>
/// <returns>
/// <see langword="true"/> to permit a reallocation request; <see langword="false"/> to make the native allocator
/// return null. The return value is ignored for shape notifications.
/// 返回 <see langword="true"/> 允许重分配；返回 <see langword="false"/> 使 native allocator 返回 null。
/// shape 通知会忽略该返回值。
/// </returns>
public delegate bool TensorRtOutputAllocatorHandler(TensorRtOutputAllocatorCallbackRequest request);
