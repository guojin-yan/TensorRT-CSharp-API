namespace JYPPX.CudaSharp;

/// <summary>
/// Captures a copied UTF-8 CUDA runtime log dump.
/// 捕获复制出的 UTF-8 CUDA runtime 日志。
/// </summary>
public sealed class CudaLogSnapshot
{
    /// <summary>Initializes a CUDA log snapshot. 初始化 CUDA 日志快照。</summary>
    public CudaLogSnapshot(string text, int bytesWritten, CudaLogCursor? nextCursor)
    {
        Text = text;
        BytesWritten = bytesWritten;
        NextCursor = nextCursor;
    }

    /// <summary>Gets the copied log text. 获取复制出的日志文本。</summary>
    public string Text { get; }

    /// <summary>Gets the number of UTF-8 bytes copied. 获取复制出的 UTF-8 字节数。</summary>
    public int BytesWritten { get; }

    /// <summary>Gets the advanced cursor when cursor-based dumping was requested. 获取使用 cursor dump 时推进后的 cursor。</summary>
    public CudaLogCursor? NextCursor { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Bytes={BytesWritten}, HasNextCursor={NextCursor.HasValue}";
}
