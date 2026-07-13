using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Describes the CUDA dependency behavior attached to a graph edge.
/// 描述 CUDA graph 边上的依赖行为。
/// </summary>
public readonly struct CudaGraphEdgeData : IEquatable<CudaGraphEdgeData>
{
    /// <summary>
    /// Initializes CUDA graph edge data.
    /// 初始化 CUDA graph 边数据。
    /// </summary>
    /// <param name="fromPort">The upstream node port. 上游节点端口。</param>
    /// <param name="toPort">The downstream node port. 下游节点端口。</param>
    /// <param name="dependencyType">The dependency type. 依赖类型。</param>
    public CudaGraphEdgeData(byte fromPort, byte toPort, CudaGraphDependencyType dependencyType)
    {
        FromPort = fromPort;
        ToPort = toPort;
        DependencyType = dependencyType;
    }

    internal CudaGraphEdgeData(NativeCudaGraphEdgeData native)
        : this(native.FromPort, native.ToPort, (CudaGraphDependencyType)native.Type)
    {
    }

    /// <summary>
    /// Gets default full-completion dependency data.
    /// 获取默认的全完成依赖数据。
    /// </summary>
    public static CudaGraphEdgeData Default => new CudaGraphEdgeData(0, 0, CudaGraphDependencyType.Default);

    /// <summary>
    /// Gets the upstream node port.
    /// 获取上游节点端口。
    /// </summary>
    public byte FromPort { get; }

    /// <summary>
    /// Gets the downstream node port.
    /// 获取下游节点端口。
    /// </summary>
    public byte ToPort { get; }

    /// <summary>
    /// Gets the dependency type.
    /// 获取依赖类型。
    /// </summary>
    public CudaGraphDependencyType DependencyType { get; }

    internal NativeCudaGraphEdgeData ToNative()
    {
        return new NativeCudaGraphEdgeData
        {
            FromPort = FromPort,
            ToPort = ToPort,
            Type = (byte)DependencyType
        };
    }

    /// <inheritdoc />
    public bool Equals(CudaGraphEdgeData other) =>
        FromPort == other.FromPort &&
        ToPort == other.ToPort &&
        DependencyType == other.DependencyType;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is CudaGraphEdgeData other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 31) + FromPort.GetHashCode();
            hash = (hash * 31) + ToPort.GetHashCode();
            hash = (hash * 31) + DependencyType.GetHashCode();
            return hash;
        }
    }

    /// <summary>
    /// Formats the edge data for diagnostics.
    /// 将 edge data 格式化为诊断字符串。
    /// </summary>
    public override string ToString() => $"FromPort={FromPort}, ToPort={ToPort}, Type={DependencyType}";

    /// <summary>
    /// Returns whether two edge data values are equal.
    /// 返回两个 edge data 是否相等。
    /// </summary>
    public static bool operator ==(CudaGraphEdgeData left, CudaGraphEdgeData right) => left.Equals(right);

    /// <summary>
    /// Returns whether two edge data values are not equal.
    /// 返回两个 edge data 是否不相等。
    /// </summary>
    public static bool operator !=(CudaGraphEdgeData left, CudaGraphEdgeData right) => !left.Equals(right);
}

/// <summary>
/// Identifies the CUDA graph dependency type encoded in graph edge data.
/// 标识 CUDA graph edge data 中的依赖类型。
/// </summary>
public enum CudaGraphDependencyType : byte
{
    /// <summary>
    /// The default full-completion dependency.
    /// 默认全完成依赖。
    /// </summary>
    Default = 0,

    /// <summary>
    /// A CUDA programmatic dependency.
    /// CUDA 程序化依赖。
    /// </summary>
    Programmatic = 1
}

/// <summary>
/// Describes a graph node reached through a dependency edge and its edge data.
/// 描述通过依赖边关联的 graph node 及其 edge data。
/// </summary>
public readonly struct CudaGraphNodeDependency
{
    /// <summary>
    /// Initializes an adjacent graph node dependency.
    /// 初始化相邻 graph node dependency。
    /// </summary>
    /// <param name="node">The adjacent node. 相邻节点。</param>
    /// <param name="edgeData">The edge data. 边数据。</param>
    public CudaGraphNodeDependency(CudaGraphNode node, CudaGraphEdgeData edgeData)
    {
        Node = node;
        EdgeData = edgeData;
    }

    /// <summary>
    /// Gets the adjacent graph node.
    /// 获取相邻 graph node。
    /// </summary>
    public CudaGraphNode Node { get; }

    /// <summary>
    /// Gets the edge data associated with the dependency.
    /// 获取该依赖对应的 edge data。
    /// </summary>
    public CudaGraphEdgeData EdgeData { get; }

    /// <summary>
    /// Formats the dependency for diagnostics.
    /// 将 dependency 格式化为诊断字符串。
    /// </summary>
    public override string ToString() => $"{Node} [{EdgeData}]";
}
