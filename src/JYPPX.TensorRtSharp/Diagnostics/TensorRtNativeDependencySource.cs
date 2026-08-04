using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Identifies how a native dependency path was discovered by <see cref="TensorRtEnvironmentProbe.ProbeNativeDependencies"/>.
/// 标识 <see cref="TensorRtEnvironmentProbe.ProbeNativeDependencies"/> 发现 native 依赖路径的方式。
/// </summary>
public enum TensorRtNativeDependencySource
{
    /// <summary>
    /// The entry is a candidate path produced by the native bridge resolver.
    /// 条目来自 native bridge resolver 生成的候选路径。
    /// </summary>
    NativeBridgeCandidate = 0,

    /// <summary>
    /// The entry is a module already loaded in the current process.
    /// 条目是当前进程中已经加载的模块。
    /// </summary>
    LoadedProcessModule = 1,

    /// <summary>
    /// The entry is a matching DLL found on the process search path.
    /// 条目是在进程搜索路径上找到的匹配 DLL。
    /// </summary>
    SearchPathCandidate = 2
}
