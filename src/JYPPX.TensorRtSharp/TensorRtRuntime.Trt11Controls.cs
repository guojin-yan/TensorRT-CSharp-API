using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtRuntime
{
    /// <summary>
    /// Gets or sets the DLA core selected for TensorRT runtime deserialization.
    /// 获取或设置 TensorRT 11 runtime 反序列化时选择的 DLA core。
    /// </summary>
    public int DlaCore
    {
        get => NativeBridgeApi.GetRuntimeDlaCore(Line, _handle);
        set => NativeBridgeApi.SetRuntimeDlaCore(Line, _handle, value);
    }

    /// <summary>
    /// Gets the number of DLA cores visible to the TensorRT runtime.
    /// 获取 TensorRT 11 runtime 可见的 DLA core 数量。
    /// </summary>
    public int DlaCoreCount => NativeBridgeApi.GetRuntimeDlaCoreCount(Line, _handle);

    /// <summary>
    /// Gets or sets the maximum number of worker threads TensorRT may use in this runtime.
    /// 获取或设置当前 runtime 允许 TensorRT 使用的最大工作线程数。
    /// </summary>
    public int MaxThreads
    {
        get => NativeBridgeApi.GetRuntimeMaxThreads(Line, _handle);
        set => NativeBridgeApi.SetRuntimeMaxThreads(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets whether TensorRT may deserialize engines that contain host executable code.
    /// 获取或设置 TensorRT 是否允许反序列化包含主机可执行代码的 engine。
    /// </summary>
    public bool EngineHostCodeAllowed
    {
        get => NativeBridgeApi.GetRuntimeEngineHostCodeAllowed(Line, _handle);
        set => NativeBridgeApi.SetRuntimeEngineHostCodeAllowed(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets TensorRT temporary-file control flags.
    /// 获取或设置 TensorRT 临时文件控制标志。
    /// </summary>
    public TensorRtTempfileControlFlags TempfileControlFlags
    {
        get => NativeBridgeApi.GetRuntimeTempfileControlFlags(Line, _handle);
        set => NativeBridgeApi.SetRuntimeTempfileControlFlags(Line, _handle, value);
    }

    /// <summary>
    /// Gets whether this runtime currently has a TensorRT error recorder attached.
    /// 获取当前 runtime 是否附加了 TensorRT error recorder。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasRuntimeErrorRecorder(Line, _handle);

    /// <summary>
    /// Gets the runtime temporary directory configured through TensorRT.
    /// 获取通过 TensorRT 配置的 runtime 临时目录。
    /// </summary>
    /// <returns>The configured temporary directory or an empty string when TensorRT uses defaults. 已配置目录；TensorRT 使用默认值时为空字符串。</returns>
    public string GetTemporaryDirectory()
    {
        return NativeBridgeApi.GetRuntimeTemporaryDirectory(Line, _handle);
    }

    /// <summary>
    /// Sets the runtime temporary directory used by TensorRT for executable temporary files.
    /// 设置 TensorRT 用于可执行临时文件的 runtime 临时目录。
    /// </summary>
    /// <param name="directoryPath">The temporary directory path. 临时目录路径。</param>
    public void SetTemporaryDirectory(string directoryPath)
    {
        NativeBridgeApi.SetRuntimeTemporaryDirectory(Line, _handle, directoryPath);
    }

    /// <summary>
    /// Clears the explicit temporary directory and lets TensorRT use platform defaults.
    /// 清除显式临时目录，让 TensorRT 使用平台默认目录。
    /// </summary>
    public void ClearTemporaryDirectory()
    {
        NativeBridgeApi.ClearRuntimeTemporaryDirectory(Line, _handle);
    }

    /// <summary>
    /// Clears the native error recorder pointer if one was attached externally.
    /// 清除外部附加的原生 error recorder 指针。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearRuntimeErrorRecorder(Line, _handle);
    }

    /// <summary>
    /// Clears the native GPU allocator pointer and returns runtime allocation to TensorRT defaults.
    /// 清除原生 GPU allocator 指针，让 runtime allocation 回到 TensorRT 默认行为。
    /// </summary>
    public void ClearGpuAllocator()
    {
        NativeBridgeApi.ClearRuntimeGpuAllocator(Line, _handle);
    }
}
