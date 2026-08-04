using System;
using System.Collections.Generic;
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
    /// 获取当前 runtime 是否附加了 TensorRT error recorder；不会暴露 recorder 指针或接管其生命周期。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasRuntimeErrorRecorder(Line, _handle);

    /// <summary>
    /// Gets whether this runtime still has a native TensorRT logger attached.
    /// 获取当前 runtime 是否仍附加 TensorRT 原生 logger；不会暴露 logger 指针或接管其生命周期。
    /// </summary>
    public bool HasLogger => NativeBridgeApi.HasRuntimeLogger(Line, _handle);

    /// <summary>
    /// Attempts to collect a copied read-only snapshot from the runtime error recorder.
    /// 尝试从 runtime error recorder 采集只读托管快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 已复制到托管内存的快照。</param>
    /// <returns><c>true</c> when a recorder was attached; otherwise <c>false</c>. 附加了 recorder 时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    /// <remarks>
    /// This method does not expose, retain, increment, decrement, or destroy the native recorder pointer.
    /// 此方法不会暴露、持有、增加引用、减少引用或销毁原生 recorder 指针。
    /// </remarks>
    public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
    {
        snapshot = NativeBridgeApi.GetRuntimeErrorRecorderSnapshot(Line, _handle);
        return snapshot.HasRecorder;
    }

    /// <summary>
    /// Gets a copied read-only diagnostic snapshot for this runtime.
    /// 获取当前 runtime 的复制型只读诊断快照。
    /// </summary>
    public TensorRtRuntimeDiagnosticSnapshot GetDiagnosticSnapshot()
    {
        List<string> diagnostics = new List<string>();
        bool hasErrorRecorder = TryCollect("HasErrorRecorder", diagnostics, () => HasErrorRecorder, false);
        TensorRtErrorRecorderSnapshot errorRecorder = TryCollect(
            "ErrorRecorderSnapshot",
            diagnostics,
            () =>
            {
                TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
                return snapshot;
            },
            new TensorRtErrorRecorderSnapshot(Line, false, 0, false, Array.Empty<TensorRtErrorRecord>()));

        return new TensorRtRuntimeDiagnosticSnapshot(
            Line,
            TryCollect("DlaCore", diagnostics, () => DlaCore, -1),
            TryCollect("DlaCoreCount", diagnostics, () => DlaCoreCount, 0),
            TryCollect("MaxThreads", diagnostics, () => MaxThreads, 0),
            TryCollect("EngineHostCodeAllowed", diagnostics, () => EngineHostCodeAllowed, false),
            TryCollect("TempfileControlFlags", diagnostics, () => TempfileControlFlags, TensorRtTempfileControlFlags.None),
            TryCollect("TemporaryDirectory", diagnostics, GetTemporaryDirectory, string.Empty),
            TryCollect("HasLogger", diagnostics, () => HasLogger, false),
            hasErrorRecorder,
            errorRecorder,
            diagnostics);
    }

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
    /// 清除外部附加的原生 error recorder 指针；不会销毁 recorder 或接管其生命周期。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearRuntimeErrorRecorder(Line, _handle);
    }

    /// <summary>
    /// Clears the native GPU allocator pointer and returns runtime allocation to TensorRT defaults.
    /// 清除原生 GPU allocator 指针，让 runtime allocation 回到 TensorRT 默认行为；不会调用用户 allocator 的 free/deallocate 回调。
    /// </summary>
    /// <remarks>
    /// A managed owner installed through <see cref="SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner)"/> remains alive while any
    /// engine deserialized by this runtime still holds its inherited borrower lease.
    /// 通过 <see cref="SetGpuAllocator(TensorRtGpuAllocatorCallbackOwner)"/> 安装的托管 owner 会持续存活，直到该 runtime
    /// 反序列化出的所有 engine 都释放继承的借用租约。
    /// </remarks>
    public void ClearGpuAllocator()
    {
        ClearManagedGpuAllocator();
    }

    private static T TryCollect<T>(string fieldName, List<string> diagnostics, Func<T> getter, T fallback)
    {
        try
        {
            return getter();
        }
        catch (Exception ex) when (ex is BridgeProbeException || ex is NotSupportedException || ex is InvalidOperationException)
        {
            diagnostics.Add($"{fieldName}: {ex.Message}");
            return fallback;
        }
    }
}
