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

    public bool Equals(CudaGraphNode other) => Token == other.Token;

    public override bool Equals(object? obj) => obj is CudaGraphNode other && Equals(other);

    public override int GetHashCode() => Token.GetHashCode();

    public override string ToString() => $"0x{Token.ToUInt64():X}";

    public static bool operator ==(CudaGraphNode left, CudaGraphNode right) => left.Equals(right);

    public static bool operator !=(CudaGraphNode left, CudaGraphNode right) => !left.Equals(right);
}
