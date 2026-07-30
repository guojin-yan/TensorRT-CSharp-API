using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Non-throwing diagnostic report for native bridge, TensorRT, parser, CUDA, and cuDNN dependency resolution.
/// 用于 native bridge、TensorRT、parser、CUDA 与 cuDNN 依赖解析的非抛异常诊断报告。
/// </summary>
public sealed class TensorRtDependencyProbeReport
{
    internal TensorRtDependencyProbeReport(
        TensorRtApiLine line,
        bool bridgeInitialized,
        string bridgeDiagnostic,
        IReadOnlyList<TensorRtNativeDependencyInfo> nativeBridgeCandidates,
        IReadOnlyList<TensorRtNativeDependencyInfo> loadedModules,
        IReadOnlyList<TensorRtNativeDependencyInfo> searchPathCandidates,
        IReadOnlyList<string> diagnostics)
    {
        Line = line;
        BridgeInitialized = bridgeInitialized;
        BridgeDiagnostic = bridgeDiagnostic ?? string.Empty;
        NativeBridgeCandidates = nativeBridgeCandidates ?? Array.Empty<TensorRtNativeDependencyInfo>();
        LoadedModules = loadedModules ?? Array.Empty<TensorRtNativeDependencyInfo>();
        SearchPathCandidates = searchPathCandidates ?? Array.Empty<TensorRtNativeDependencyInfo>();
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to choose expected TensorRT DLL names.
    /// 获取用于选择预期 TensorRT DLL 名称的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the managed native bridge loader initialized successfully.
    /// 获取托管 native bridge loader 是否已成功初始化。
    /// </summary>
    public bool BridgeInitialized { get; }

    /// <summary>
    /// Gets the bridge loader diagnostic message.
    /// 获取 bridge loader 的诊断消息。
    /// </summary>
    public string BridgeDiagnostic { get; }

    /// <summary>
    /// Gets native bridge candidate paths produced by the same resolver used for DllImport loading.
    /// 获取由 DllImport 加载使用的同一 resolver 生成的 native bridge 候选路径。
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> NativeBridgeCandidates { get; }

    /// <summary>
    /// Gets matching modules that are already loaded in the current process.
    /// 获取当前进程中已经加载的匹配模块。
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> LoadedModules { get; }

    /// <summary>
    /// Gets matching dependency DLLs found on the current process search path.
    /// 获取在当前进程搜索路径上找到的匹配依赖 DLL。
    /// </summary>
    public IReadOnlyList<TensorRtNativeDependencyInfo> SearchPathCandidates { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while probing modules and directories.
    /// 获取探测模块和目录时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Gets the number of matching loaded modules.
    /// 获取匹配的已加载模块数量。
    /// </summary>
    public int LoadedModuleCount => LoadedModules.Count;

    /// <summary>
    /// Gets the number of matching search-path candidates.
    /// 获取匹配的搜索路径候选项数量。
    /// </summary>
    public int SearchPathCandidateCount => SearchPathCandidates.Count;

    /// <summary>
    /// Formats the dependency probe report for diagnostics.
    /// 将依赖探针报告格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"{Line}:bridge={BridgeInitialized}:loaded={LoadedModuleCount}:pathCandidates={SearchPathCandidateCount}:diagnostics={Diagnostics.Count}";
    }
}
