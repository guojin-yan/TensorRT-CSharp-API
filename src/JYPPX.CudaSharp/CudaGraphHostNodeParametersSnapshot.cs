using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures callback-presence metadata without exposing a CUDA host callback or user-data pointer.
/// 捕获 callback 存在性元数据，不暴露 CUDA host callback 或 user-data pointer。
/// </summary>
public sealed class CudaGraphHostNodeParametersSnapshot
{
    internal CudaGraphHostNodeParametersSnapshot(NativeCudaGraphHostNodeParamsSnapshot native)
    {
        HasCallback = native.HasCallback != 0;
        HasUserData = native.HasUserData != 0;
    }

    /// <summary>Gets whether the native host node had a callback. 获取 native host 节点是否包含 callback。</summary>
    public bool HasCallback { get; }
    /// <summary>Gets whether the native host node had user data. 获取 native host 节点是否包含用户数据。</summary>
    public bool HasUserData { get; }

    /// <inheritdoc />
    public override string ToString() => $"HasCallback={HasCallback}, HasUserData={HasUserData}";
}
