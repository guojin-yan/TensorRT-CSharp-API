using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies a position in CUDA's runtime log buffer.
/// 标识 CUDA runtime 日志缓冲区中的位置。
/// </summary>
public readonly struct CudaLogCursor : IEquatable<CudaLogCursor>
{
    internal CudaLogCursor(uint value)
    {
        Value = value;
    }

    internal uint Value { get; }

    /// <summary>Compares two CUDA log cursors. 比较两个 CUDA log cursor。</summary>
    public bool Equals(CudaLogCursor other) => Value == other.Value;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is CudaLogCursor other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => Value.GetHashCode();

    /// <inheritdoc />
    public override string ToString() => Value.ToString();

    /// <summary>Tests two CUDA log cursors for equality. 测试两个 CUDA log cursor 是否相等。</summary>
    public static bool operator ==(CudaLogCursor left, CudaLogCursor right) => left.Equals(right);

    /// <summary>Tests two CUDA log cursors for inequality. 测试两个 CUDA log cursor 是否不相等。</summary>
    public static bool operator !=(CudaLogCursor left, CudaLogCursor right) => !left.Equals(right);
}
