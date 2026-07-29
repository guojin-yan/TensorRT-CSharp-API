using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Copied optional CUDA Driver capability snapshot. 可选 CUDA Driver 能力的复制型快照。</summary>
public sealed class CudaDriverCapability
{
    internal CudaDriverCapability(NativeCudaDriverCapabilityInfo info, string loadedLibraryName, string dependencyDiagnostic)
    {
        IsAvailable = info.DependencyAvailable != 0;
        DriverVersion = info.DriverVersion;
        SupportsModuleLoad = info.SupportsModuleLoad != 0;
        SupportsFunctionLookup = info.SupportsFunctionLookup != 0;
        SupportsTypedLaunch = info.SupportsTypedLaunch != 0;
        SupportsContextInterop = info.SupportsContextInterop != 0;
        SupportsCompletionEvents = info.SupportsCompletionEvents != 0;
        LoadedLibraryName = loadedLibraryName;
        DependencyDiagnostic = dependencyDiagnostic;
    }

    /// <summary>Gets whether the optional CUDA Driver library initialized successfully. 获取可选 CUDA Driver library 是否成功初始化。</summary>
    public bool IsAvailable { get; }

    /// <summary>Gets the encoded CUDA Driver version. 获取编码后的 CUDA Driver version。</summary>
    public int DriverVersion { get; }

    /// <summary>Gets whether Driver module loading is available. 获取是否支持 Driver module loading。</summary>
    public bool SupportsModuleLoad { get; }

    /// <summary>Gets whether named function lookup is available inside the bridge. 获取 bridge 内是否支持按名称 function lookup。</summary>
    public bool SupportsFunctionLookup { get; }

    /// <summary>Gets whether typed kernel launch is available. 获取是否支持 typed kernel launch。</summary>
    public bool SupportsTypedLaunch { get; }

    /// <summary>Gets whether context push/pop interop is available. 获取是否支持 context push/pop interop。</summary>
    public bool SupportsContextInterop { get; }

    /// <summary>Gets whether completion events are available. 获取是否支持 completion events。</summary>
    public bool SupportsCompletionEvents { get; }

    /// <summary>Gets the actual dynamically loaded library name. 获取实际动态加载的 library 名称。</summary>
    public string LoadedLibraryName { get; }

    /// <summary>Gets an actionable dependency diagnostic when unavailable. 获取不可用时的可操作依赖诊断。</summary>
    public string DependencyDiagnostic { get; }
}

/// <summary>High-level optional CUDA Driver capability entry point. 可选 CUDA Driver 能力的高层入口。</summary>
public static class CudaDriver
{
    /// <summary>Queries the CUDA Driver without making it a bridge load-time dependency. 查询 CUDA Driver，但不把它变成 bridge 加载时依赖。</summary>
    public static CudaDriverCapability GetCapability()
    {
        NativeBridgeLoader.EnsureInitialized();
        return new CudaDriverCapability(
            NativeCudaApi.QueryDriverCapability(),
            NativeCudaApi.GetDriverLoadedLibraryName(),
            NativeCudaApi.GetDriverDependencyDiagnostic());
    }
}
