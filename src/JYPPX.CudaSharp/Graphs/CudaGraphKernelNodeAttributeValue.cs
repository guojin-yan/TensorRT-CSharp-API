using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies CUDA graph kernel node attributes supported by the safe scalar bridge.
/// 标识安全标量 bridge 支持的 CUDA graph kernel node attribute。
/// </summary>
public enum CudaGraphKernelNodeAttribute
{
    /// <summary>
    /// Cooperative launch flag.
    /// Cooperative launch 标志。
    /// </summary>
    Cooperative = 2,

    /// <summary>
    /// Cluster dimension.
    /// Cluster 维度。
    /// </summary>
    ClusterDimension = 4,

    /// <summary>
    /// Cluster scheduling policy preference.
    /// Cluster 调度策略偏好。
    /// </summary>
    ClusterSchedulingPolicyPreference = 5,

    /// <summary>
    /// Kernel execution priority.
    /// Kernel 执行优先级。
    /// </summary>
    Priority = 8
}

/// <summary>
/// Describes CUDA cluster scheduling policy preference values.
/// 描述 CUDA cluster scheduling policy preference 值。
/// </summary>
public enum CudaClusterSchedulingPolicyPreference
{
    /// <summary>
    /// Default CUDA policy.
    /// CUDA 默认策略。
    /// </summary>
    Default = 0,

    /// <summary>
    /// Spread blocks within a cluster to SMs.
    /// 将 cluster 内的 block 分散到 SM。
    /// </summary>
    Spread = 1,

    /// <summary>
    /// Allow hardware load balancing.
    /// 允许硬件负载均衡。
    /// </summary>
    LoadBalancing = 2
}

/// <summary>
/// Describes a CUDA graph kernel node attribute using a copied scalar descriptor.
/// 使用复制出的标量 descriptor 描述 CUDA graph kernel node attribute。
/// </summary>
public readonly struct CudaGraphKernelNodeAttributeValue
{
    private CudaGraphKernelNodeAttributeValue(CudaGraphKernelNodeAttribute attribute, int intValue, uint x, uint y, uint z)
    {
        Attribute = attribute;
        IntValue = intValue;
        X = x;
        Y = y;
        Z = z;
    }

    internal CudaGraphKernelNodeAttributeValue(NativeCudaGraphKernelNodeAttributeValue native)
        : this((CudaGraphKernelNodeAttribute)native.Attribute, native.IntValue, native.X, native.Y, native.Z)
    {
    }

    /// <summary>
    /// Gets the attribute represented by this descriptor.
    /// 获取该 descriptor 表示的 attribute。
    /// </summary>
    public CudaGraphKernelNodeAttribute Attribute { get; }

    /// <summary>
    /// Gets the scalar integer value for cooperative, priority, or policy attributes.
    /// 获取 cooperative、priority 或 policy attribute 的标量整数值。
    /// </summary>
    public int IntValue { get; }

    /// <summary>
    /// Gets the X component for cluster dimension attributes.
    /// 获取 cluster dimension attribute 的 X 分量。
    /// </summary>
    public uint X { get; }

    /// <summary>
    /// Gets the Y component for cluster dimension attributes.
    /// 获取 cluster dimension attribute 的 Y 分量。
    /// </summary>
    public uint Y { get; }

    /// <summary>
    /// Gets the Z component for cluster dimension attributes.
    /// 获取 cluster dimension attribute 的 Z 分量。
    /// </summary>
    public uint Z { get; }

    /// <summary>
    /// Gets whether the cooperative attribute is enabled.
    /// 获取 cooperative attribute 是否启用。
    /// </summary>
    public bool CooperativeEnabled => Attribute == CudaGraphKernelNodeAttribute.Cooperative && IntValue != 0;

    /// <summary>
    /// Gets the cluster scheduling policy preference.
    /// 获取 cluster scheduling policy preference。
    /// </summary>
    public CudaClusterSchedulingPolicyPreference ClusterSchedulingPolicyPreference => (CudaClusterSchedulingPolicyPreference)IntValue;

    /// <summary>
    /// Creates a cooperative kernel node attribute descriptor.
    /// 创建 cooperative kernel node attribute descriptor。
    /// </summary>
    /// <param name="enabled">Whether cooperative launch is enabled. 是否启用 cooperative launch。</param>
    public static CudaGraphKernelNodeAttributeValue Cooperative(bool enabled) =>
        new(CudaGraphKernelNodeAttribute.Cooperative, enabled ? 1 : 0, 0, 0, 0);

    /// <summary>
    /// Creates a priority kernel node attribute descriptor.
    /// 创建 priority kernel node attribute descriptor。
    /// </summary>
    /// <param name="priority">The kernel execution priority. Kernel 执行优先级。</param>
    public static CudaGraphKernelNodeAttributeValue Priority(int priority) =>
        new(CudaGraphKernelNodeAttribute.Priority, priority, 0, 0, 0);

    /// <summary>
    /// Creates a cluster dimension kernel node attribute descriptor.
    /// 创建 cluster dimension kernel node attribute descriptor。
    /// </summary>
    /// <param name="x">The cluster X dimension. Cluster X 维度。</param>
    /// <param name="y">The cluster Y dimension. Cluster Y 维度。</param>
    /// <param name="z">The cluster Z dimension. Cluster Z 维度。</param>
    public static CudaGraphKernelNodeAttributeValue ClusterDimension(uint x, uint y, uint z)
    {
        if (x == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(x), "Cluster dimensions must be positive.");
        }

        if (y == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(y), "Cluster dimensions must be positive.");
        }

        if (z == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(z), "Cluster dimensions must be positive.");
        }

        return new CudaGraphKernelNodeAttributeValue(CudaGraphKernelNodeAttribute.ClusterDimension, 0, x, y, z);
    }

    /// <summary>
    /// Creates a cluster scheduling policy preference descriptor.
    /// 创建 cluster scheduling policy preference descriptor。
    /// </summary>
    /// <param name="policy">The policy preference. 策略偏好。</param>
    public static CudaGraphKernelNodeAttributeValue ClusterSchedulingPolicy(CudaClusterSchedulingPolicyPreference policy)
    {
        if ((int)policy < 0 || (int)policy > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(policy));
        }

        return new CudaGraphKernelNodeAttributeValue(CudaGraphKernelNodeAttribute.ClusterSchedulingPolicyPreference, (int)policy, 0, 0, 0);
    }

    internal NativeCudaGraphKernelNodeAttributeValue ToNative() =>
        new()
        {
            Attribute = (int)Attribute,
            IntValue = IntValue,
            X = X,
            Y = Y,
            Z = Z
        };

    /// <summary>
    /// Formats the copied kernel node attribute descriptor for diagnostics.
    /// 将复制出的 kernel node attribute descriptor 格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        Attribute == CudaGraphKernelNodeAttribute.ClusterDimension
            ? $"{Attribute}: {X}x{Y}x{Z}"
            : $"{Attribute}: {IntValue}";
}
