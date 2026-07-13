namespace JYPPX.CudaSharp;

/// <summary>
/// Describes the CUDA memcpy direction for copied diagnostics.
/// 描述复制诊断信息中的 CUDA memcpy 方向。
/// </summary>
public enum CudaMemcpyKind
{
    /// <summary>
    /// Host-to-device copy.
    /// Host 到 device 复制。
    /// </summary>
    HostToDevice = 1,

    /// <summary>
    /// Device-to-host copy.
    /// Device 到 host 复制。
    /// </summary>
    DeviceToHost = 2,

    /// <summary>
    /// Device-to-device copy.
    /// Device 到 device 复制。
    /// </summary>
    DeviceToDevice = 3
}
