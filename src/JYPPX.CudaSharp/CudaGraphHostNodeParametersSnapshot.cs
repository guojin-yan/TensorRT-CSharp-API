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

    public bool HasCallback { get; }
    public bool HasUserData { get; }

    /// <inheritdoc />
    public override string ToString() => $"HasCallback={HasCallback}, HasUserData={HasUserData}";
}
