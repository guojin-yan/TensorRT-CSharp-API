using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies the kind of a CUDA managed-memory location.
/// 标识 CUDA managed memory 位置的种类。
/// </summary>
public enum CudaMemoryLocationKind
{
    /// <summary>
    /// A CUDA device identified by its ordinal. 由设备序号标识的 CUDA 设备。
    /// </summary>
    Device = 1,

    /// <summary>
    /// Host memory without a specific NUMA node. 不指定 NUMA 节点的主机内存。
    /// </summary>
    Host = 2,

    /// <summary>
    /// A specific host NUMA node. 指定的主机 NUMA 节点。
    /// </summary>
    HostNuma = 3,

    /// <summary>
    /// The host NUMA node nearest to the current thread. 最接近当前线程的主机 NUMA 节点。
    /// </summary>
    CurrentHostNuma = 4
}

/// <summary>
/// Represents a validated, pointer-free CUDA managed-memory location.
/// 表示经过校验且不包含指针的 CUDA managed memory 位置。
/// </summary>
public readonly struct CudaMemoryLocation : IEquatable<CudaMemoryLocation>
{
    private CudaMemoryLocation(CudaMemoryLocationKind kind, int id)
    {
        Kind = kind;
        Id = id;
    }

    /// <summary>
    /// Gets the location kind. 获取位置种类。
    /// </summary>
    public CudaMemoryLocationKind Kind { get; }

    /// <summary>
    /// Gets the CUDA device ordinal or host NUMA node identifier when applicable.
    /// 获取适用时的 CUDA 设备序号或主机 NUMA 节点标识。
    /// </summary>
    public int Id { get; }

    /// <summary>
    /// Gets the canonical host-memory location. 获取规范化的主机内存位置。
    /// </summary>
    public static CudaMemoryLocation Host { get; } = new CudaMemoryLocation(CudaMemoryLocationKind.Host, 0);

    /// <summary>
    /// Gets the canonical location for the host NUMA node nearest to the current thread.
    /// 获取最接近当前线程的主机 NUMA 节点规范位置。
    /// </summary>
    public static CudaMemoryLocation CurrentHostNuma { get; } = new CudaMemoryLocation(CudaMemoryLocationKind.CurrentHostNuma, 0);

    /// <summary>
    /// Creates a CUDA device location. 创建 CUDA 设备位置。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    public static CudaMemoryLocation Device(int deviceOrdinal)
    {
        if (deviceOrdinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceOrdinal));
        }

        return new CudaMemoryLocation(CudaMemoryLocationKind.Device, deviceOrdinal);
    }

    /// <summary>
    /// Creates a specific host NUMA-node location. 创建指定的主机 NUMA 节点位置。
    /// </summary>
    /// <param name="numaNodeId">The non-negative host NUMA node identifier. 非负的主机 NUMA 节点标识。</param>
    public static CudaMemoryLocation HostNuma(int numaNodeId)
    {
        if (numaNodeId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numaNodeId));
        }

        return new CudaMemoryLocation(CudaMemoryLocationKind.HostNuma, numaNodeId);
    }

    /// <inheritdoc />
    public bool Equals(CudaMemoryLocation other)
    {
        return Kind == other.Kind && Id == other.Id;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        return obj is CudaMemoryLocation other && Equals(other);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        unchecked
        {
            return ((int)Kind * 397) ^ Id;
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return Kind == CudaMemoryLocationKind.Device || Kind == CudaMemoryLocationKind.HostNuma
            ? $"{Kind}:{Id}"
            : Kind.ToString();
    }

    /// <summary>
    /// Compares two CUDA memory locations. 比较两个 CUDA memory 位置。
    /// </summary>
    public static bool operator ==(CudaMemoryLocation left, CudaMemoryLocation right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Compares two CUDA memory locations. 比较两个 CUDA memory 位置。
    /// </summary>
    public static bool operator !=(CudaMemoryLocation left, CudaMemoryLocation right)
    {
        return !left.Equals(right);
    }

    internal void Validate(string parameterName)
    {
        bool valid = Kind switch
        {
            CudaMemoryLocationKind.Device or CudaMemoryLocationKind.HostNuma => Id >= 0,
            CudaMemoryLocationKind.Host or CudaMemoryLocationKind.CurrentHostNuma => Id == 0,
            _ => false
        };

        if (!valid)
        {
            throw new ArgumentException("The CUDA memory location is invalid or non-canonical.", parameterName);
        }
    }
}
