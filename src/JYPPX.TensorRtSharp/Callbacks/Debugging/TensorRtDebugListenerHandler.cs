namespace JYPPX.TensorRtSharp;

/// <summary>
/// Handles one TensorRT debug-tensor notification using copied, pointer-free metadata.
/// 使用复制且不含指针的元数据处理一次 TensorRT debug-tensor 通知。
/// </summary>
/// <param name="metadata">Copied tensor metadata. 复制出的 tensor 元数据。</param>
/// <returns><see langword="true"/> on success; otherwise TensorRT receives callback failure. 成功时返回 true，否则 TensorRT 会收到 callback failure。</returns>
public delegate bool TensorRtDebugListenerHandler(TensorRtDebugTensorMetadataSnapshot metadata);
