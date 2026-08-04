namespace JYPPX.CudaSharp;

/// <summary>
/// Repository-level metadata for the managed CUDA assembly.
/// 托管 CUDA 程序集的仓库级元数据。
/// </summary>
public static class CudaSharpInfo
{
    /// <summary>
    /// The native bridge logical library name.
    /// native bridge 的逻辑库名称。
    /// </summary>
    public static string NativeBridgeLibraryName => JYPPX.TensorRtSharp.Shared.BridgeConstants.NativeBridgeLibraryName;
}
