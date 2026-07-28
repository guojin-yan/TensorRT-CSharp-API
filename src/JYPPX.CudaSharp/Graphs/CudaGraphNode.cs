using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies a CUDA graph node owned by a <see cref="CudaGraph"/>.
/// 标识一个由 <see cref="CudaGraph"/> 拥有的 CUDA graph node。
/// </summary>
/// <remarks>
/// The token is graph-owned and must not be destroyed independently.
/// 该 token 由 graph 拥有，不能被单独销毁。
/// </remarks>
public readonly struct CudaGraphNode : IEquatable<CudaGraphNode>
{
    internal CudaGraphNode(UIntPtr token)
    {
        Token = token;
    }

    internal UIntPtr Token { get; }

    /// <summary>
    /// Gets whether this node token is empty.
    /// 获取当前 node token 是否为空。
    /// </summary>
    public bool IsNull => Token == UIntPtr.Zero;

    /// <summary>
    /// Compares two node tokens for equality.
    /// 比较两个 node token 是否相等。
    /// </summary>
    /// <param name="other">The other node token. 另一个 node token。</param>
    /// <returns><see langword="true"/> when both tokens identify the same node. 当两个 token 标识同一节点时返回 <see langword="true"/>。</returns>
    public bool Equals(CudaGraphNode other) => Token == other.Token;

    /// <summary>
    /// Compares the current node token with another object.
    /// 将当前 node token 与另一个对象进行比较。
    /// </summary>
    /// <param name="obj">The object to compare. 要比较的对象。</param>
    /// <returns><see langword="true"/> when the object is an equal <see cref="CudaGraphNode"/>. 当对象是相等的 <see cref="CudaGraphNode"/> 时返回 <see langword="true"/>。</returns>
    public override bool Equals(object? obj) => obj is CudaGraphNode other && Equals(other);

    /// <summary>
    /// Returns a hash code for the node token.
    /// 返回当前 node token 的哈希代码。
    /// </summary>
    public override int GetHashCode() => Token.GetHashCode();

    /// <summary>
    /// Formats the node token as hexadecimal text.
    /// 将 node token 格式化为十六进制文本。
    /// </summary>
    public override string ToString() => $"0x{Token.ToUInt64():X}";

    /// <summary>
    /// Returns whether two node tokens are equal.
    /// 返回两个 node token 是否相等。
    /// </summary>
    public static bool operator ==(CudaGraphNode left, CudaGraphNode right) => left.Equals(right);

    /// <summary>
    /// Returns whether two node tokens are not equal.
    /// 返回两个 node token 是否不相等。
    /// </summary>
    public static bool operator !=(CudaGraphNode left, CudaGraphNode right) => !left.Equals(right);
}
