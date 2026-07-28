namespace JYPPX.CudaSharp;

/// <summary>
/// Selects a CUDA kernel attribute that cudaKernelSetAttributeForDevice permits callers to modify.
/// 选择 cudaKernelSetAttributeForDevice 允许调用方修改的 CUDA kernel 属性。
/// </summary>
public enum CudaKernelAttribute
{
    /// <summary>Sets the maximum dynamic shared-memory size in bytes. 设置最大动态 shared memory 字节数。</summary>
    MaxDynamicSharedMemorySize = 8,

    /// <summary>Sets the preferred shared-memory carveout percentage. 设置首选 shared-memory carveout 百分比。</summary>
    PreferredSharedMemoryCarveout = 9,

    /// <summary>Sets the required cluster width. 设置必需的 cluster width。</summary>
    RequiredClusterWidth = 11,

    /// <summary>Sets the required cluster height. 设置必需的 cluster height。</summary>
    RequiredClusterHeight = 12,

    /// <summary>Sets the required cluster depth. 设置必需的 cluster depth。</summary>
    RequiredClusterDepth = 13,

    /// <summary>Controls whether non-portable cluster sizes are allowed. 控制是否允许 non-portable cluster size。</summary>
    NonPortableClusterSizeAllowed = 14,

    /// <summary>Sets the cluster scheduling-policy preference. 设置 cluster scheduling policy 倾向。</summary>
    ClusterSchedulingPolicyPreference = 15
}
